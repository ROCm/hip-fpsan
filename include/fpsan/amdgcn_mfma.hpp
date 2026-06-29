// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/amdgcn_mfma.hpp
// ----------------------------------------------------------------------------
// FPSan wrappers for AMDGPU MFMA (Matrix Fused-Multiply-Add) intrinsics --
// the gfx9-family wave64 matrix instruction set, including CDNA4 (gfx950 /
// MI350)'s 16x16x32 and 32x32x16 shapes, F64 MFMAs, and the new
// 16x16x128 / 32x32x64 f8f6f4 scaled MFMAs.
//
// As with amdgcn_matrix.hpp (WMMA), Native mode forwards each wrapper to the
// real __builtin_amdgcn_mfma_* and FPSan mode runs a wave-cooperative
// software MMA in the payload ring, using the same fragment layout the
// hardware uses. The layout helpers
//
//   input_loc(dim, K, B, i, k, b, data_bits)  ->  (reg, lane, sub)
//   output_loc_32(M, N, i, j, b)              ->  (reg, lane)
//   output_loc_64(M, N, i, j, b)              ->  (reg, lane)
//
// are pinned to real gfx950 silicon as the SOURCE OF TRUTH: every shape has a
// LayoutMatchesHardware test that compares the software dataflow's gather to
// the actual __builtin_amdgcn_mfma_* on-device, so the silicon behaviour wins
// for every shape (including the f64 4x4x4 shape). Nothing here is tuned to
// match any reference other than the hardware.
//
// gfx9-family AccMode is "Unified" (output register == input register file),
// matching how CDNA3 and CDNA4 schedule MFMA -- so a single dataflow template
// covers every MFMA shape; only the M/N/K/B numbers and the element types
// vary.
//
// Wave64.  HIP/device-only.  Opt-in (not pulled by <fpsan/fpsan.hpp>).
// ----------------------------------------------------------------------------
#ifndef FPSAN_AMDGCN_MFMA_HPP
#define FPSAN_AMDGCN_MFMA_HPP

#include "fpsan/amdgcn_matrix.hpp" // detail::v8_fragment + native vec aliases
#include "fpsan/cast.hpp"
#include "fpsan/detail/fp8.hpp"
#include "fpsan/detail/native_vec.hpp"
#include "fpsan/detail/subbyte_widen.hpp"
#include "fpsan/value.hpp"

#include <cstdint>
#include <type_traits>

#if !defined(__HIP__) && !defined(__CUDACC__)
#error "fpsan/amdgcn_mfma.hpp is GPU-only; compile as HIP (or CUDA)."
#endif

namespace fpsan {

// =============================================================================
// Native vector aliases for the MFMA builtin ABIs. Names mirror the LLVM
// shorthand: v<N><T>_native where N is the per-lane element count and T is
// the element type.
// =============================================================================
using v2f_native = float __attribute__((ext_vector_type(2)));
using v16f_native = float __attribute__((ext_vector_type(16)));
using v32f_native = float __attribute__((ext_vector_type(32)));
using v4d_native = double __attribute__((ext_vector_type(4)));
// Legacy gfx9 f16 / bf16_1k MFMAs take a 4-element input fragment per lane.
using v4h_native = _Float16 __attribute__((ext_vector_type(4)));
using v4bf_native = __bf16 __attribute__((ext_vector_type(4)));
// The f8f6f4 scaled-MFMA builtins take their 32-byte-per-lane A/B operands as
// an 8-element i32 vector (the LLVM ABI for the packed fp8/fp6/fp4 fragment).
using v8i32_native = int __attribute__((ext_vector_type(8)));
// v4f_native / v8h_native / v8f_native / v8bf_native come from amdgcn_matrix.hpp.

// CDNA4 fp8 MFMA: A and B are packed as a long int (8 fp8 bytes per lane).
// LLVM's signature uses 'Wi' = i64. We expose the wrappers via Value<i64>
// over the storage type the customer is already using -- callers bit_cast
// their packed-byte register into an i64 and hand it to the wrapper.

// f8f6f4 scaled MFMA fragment: 32 bytes per lane (8 i32 worth of packed
// fp8/fp6/fp4 bytes).  Value<...>'s template param has to be a recognized
// fp type, so we wrap 32 fp8_e4m3 bytes in a v32_fragment.  The per-byte
// interpretation (fp8 / bf8 / fp6 / bf6 / fp4) is selected by the scale-op
// immediate at the call site -- the byte storage is opaque to the wrapper.
namespace detail {
// The 32-byte scaled-MFMA operand and the 16-byte fp8/bf8 SMFMAC operand
// (used by the gfx950 fp8 SMFMAC A operand and the CDNA3 fp8 SMFMAC B
// operand). Same POD shape as detail::v8_fragment in amdgcn_matrix.hpp;
// Value's vector path treats them as a 32- / 16-lane u8.
template <class Elem> using v32_fragment = vec_fragment<Elem, 32>;
template <class Elem> using v16_fragment = vec_fragment<Elem, 16>;
} // namespace detail
using v32e4m3_native = detail::v32_fragment<fp8_e4m3>;
using v32e5m2_native = detail::v32_fragment<fp8_e5m2>; // bf8 scaled-MFMA operand
using v16e4m3_native = detail::v16_fragment<fp8_e4m3>;
using v16e5m2_native = detail::v16_fragment<fp8_e5m2>;
using v8amd_e4m3_native = detail::v8_fragment<amd_fp8_e4m3>;
using v8amd_e5m2_native = detail::v8_fragment<amd_fp8_e5m2>;
using v16amd_e4m3_native = detail::v16_fragment<amd_fp8_e4m3>;
using v16amd_e5m2_native = detail::v16_fragment<amd_fp8_e5m2>;

namespace detail {

// Wave64 lane id and cross-lane shuffle come from amdgcn_matrix.hpp:
// wave_lane_full() is valid for any wavefront size (mbcnt_hi continues the
// population count into the high half of the exec mask), and wave_shfl()
// moves a scalar between lanes via ds_bpermute -- both already 64-lane
// correct, so the MFMA/SMFMAC paths reuse them directly.

// ---- gfx9 MFMA fragment layout ---------------------------------------------
// input_loc / output_loc_32 / output_loc_64 for the GFX9 (CDNA) MFMA register
// layout. Each shape that uses these is pinned to real gfx950 silicon by a
// LayoutMatchesHardware test; the 4x4x4 shape, where the dense formula does
// not apply, is handled separately.
//
//   dim         outer dim of the matrix (M for A, N for B)
//   K           reduction dimension
//   B           number of blocks
//   i           outer index (row for A, column for B)
//   k           reduction index
//   b           block index
//   data_bits   element size in bits (8, 16, 32, 64)
//
// Returns: lane that holds the element, the per-lane register slot (in
// "register" = 32-bit dword) carrying it, and which sub-element within that
// dword (0..per_dword-1).
struct InputLoc {
  int reg;
  int lane;
  int sub; // sub-element within the dword (0 for 32/64-bit elements)
};
FPSAN_HOST_DEVICE inline InputLoc input_loc(int dim, int K, int B, int i, int k, int b,
                                            int data_bits) {
  const int lanes_per_block = 64 / (dim * B);
  // elems_per_group is 0 when K < lanes_per_block (e.g. the 4x4x4 shape, whose
  // K=4 < 16 lanes/block). Those shapes don't use this dense layout at all, but
  // guard against the k % 0 / k / 0 UB so a stray call is at least defined.
  const int elems_per_group_raw = K / lanes_per_block;
  const int elems_per_group = elems_per_group_raw ? elems_per_group_raw : 1;
  const int local = k % elems_per_group;
  const int lane = b * dim + (k / elems_per_group) * dim * B + i;
  if (data_bits == 64)
    return {local * 2, lane, 0};
  if (data_bits == 32)
    return {local, lane, 0};
  const int per_dword = 32 / data_bits;
  return {local / per_dword, lane, local % per_dword};
}

struct OutputLoc {
  int reg;
  int lane;
};
FPSAN_HOST_DEVICE inline OutputLoc output_loc_32(int M, int N, int i, int j, int b) {
  const int multirows = 64 / N;
  const int mn_div_4 = (M * N) / 4;
  const int blocks_per_reg = (64 + mn_div_4 - 1) / mn_div_4;
  const int reg = b * ((M * N) / 64) + (i / (4 * multirows)) * 4 + (i % 4);
  const int lane = (b % blocks_per_reg) * N + ((i / 4) % multirows) * blocks_per_reg * N + j;
  return {reg, lane};
}

FPSAN_HOST_DEVICE inline OutputLoc output_loc_64(int M, int N, int i, int j, int b) {
  const int multirows = 64 / N;
  const int mn = M * N;
  const int blocks_per_reg = (mn > 0) ? (64 + mn - 1) / mn : 1;
  const int local = b * (mn / 64) + (i / multirows);
  const int lane = (b % blocks_per_reg) * N + (i % multirows) * blocks_per_reg * N + j;
  return {local * 2, lane};
}

FPSAN_DEVICE inline int mfma_a_src_lane(int lane, int cbsz, int abid) {
  if (cbsz == 0)
    return lane;
  const int block = 64 / (1 << cbsz);
  return (lane % block) + block * abid;
}

FPSAN_DEVICE inline int mfma_b_src_lane(int lane, int blgp) {
  switch (blgp) {
  case 0:
    return lane;
  case 1:
    return lane % 32;
  case 2:
    return (lane % 32) + 32;
  case 3:
    return (lane + 16) % 64;
  case 4:
    return lane % 16;
  case 5:
    return (lane % 16) + 16;
  case 6:
    return (lane % 16) + 32;
  case 7:
    return (lane % 16) + 48;
  default:
    return lane;
  }
}

// ---- Wave-cooperative software MFMA ----------------------------------------
// Generic over: (M, N, K, B), element types of A (AElem) and B (BElem), and
// the accumulator/output element type (CElem). All three Values carry per-
// lane fragments; the dataflow walks the K dimension, gathers A[i][k] and
// B[k][j] from the lanes that hold them, multiplies and accumulates in the
// payload ring (FPSan) or in real float (Native mode -- used only as an
// oracle in tests; production Native mode goes straight to the builtin).
//
// The lane mapping comes from input_loc / output_loc_32 (or _64 for f64
// accumulators). InRegA / InRegB select the per-lane scalar at a given
// (reg, sub) within the A/B fragment.
//
// Caller must supply two extractor lambdas:
//   ExtractA(reg_idx, sub_idx) -> scalar AElem Value (from this lane's
//                                 fragment, before shuffle)
//   ExtractB(reg_idx, sub_idx) -> scalar BElem Value
//
// The dataflow shuffles them across lanes and accumulates.  For pure-
// vector fragments (v8h, v8bf, v4f), ExtractA is simply
//   a.get(reg_or_combined_index)
// For packed (v2i32 / long int) fragments, the extractor unpacks the byte.
template <int M, int N, int K, int B, int InBits, class CElem, Semantics S, Conversions C,
          class AFrag, class BFrag, class CFrag, class ExtractA, class ExtractB>
FPSAN_DEVICE CFrag mfma_software(AFrag a, BFrag b, CFrag c, ExtractA ea, ExtractB eb, int cbsz,
                                 int abid, int blgp);

template <int M, int N, int K, int B, int InBits, class CElem, Semantics S, Conversions C,
          class AFrag, class BFrag, class CFrag, class ExtractA, class ExtractB>
FPSAN_DEVICE CFrag mfma_software(AFrag a, BFrag b, CFrag c, ExtractA ea, ExtractB eb) {
  return mfma_software<M, N, K, B, InBits, CElem, S, C>(a, b, c, ea, eb, 0, 0, 0);
}

template <int M, int N, int K, int B, int InBits, class CElem, Semantics S, Conversions C,
          class AFrag, class BFrag, class CFrag, class ExtractA, class ExtractB>
FPSAN_DEVICE CFrag mfma_software(AFrag a, BFrag b, CFrag c, ExtractA ea, ExtractB eb, int cbsz,
                                 int abid, int blgp) {
  (void)a;
  (void)b;
  using Acc = Value<CElem, S, C>;
  const int lane = wave_lane_full();
  // Per-lane register count for the output fragment:
  //   M*N*B / 64 elements per lane.
  constexpr int per_lane = (M * N * B) / 64;
  // The output register layout depends on the accumulator element width: 64-bit
  // (f64) accumulators occupy VGPR *pairs*, so they use output_loc_64 (whose
  // `reg` counts 32-bit dwords -> the fragment slot is reg/2). 32-bit (f32)
  // accumulators use output_loc_32 (reg == fragment slot). This split matches
  // the hardware and is pinned by the MfmaF64_16x16x4.LayoutMatchesHardware
  // test, which fails if f64 is forced through output_loc_32.
  constexpr bool acc64 = sizeof(CElem) == 8;
  CFrag d{};
  // For each output reg this lane carries, derive the (i, j, b) it holds,
  // then sum over K.
  for (int reg = 0; reg < per_lane; ++reg) {
    // Invert the output layout: iterate over (i, j, b) and pick the one whose
    // forward map lands on this lane's fragment slot `reg`. The forward map is
    // closed-form; inverting in closed form would be fragile. Cheap because
    // each lane has at most a few hits.
    int oi = -1, oj = -1, ob = -1;
    for (int bb = 0; bb < B && ob < 0; ++bb) {
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          OutputLoc loc = acc64 ? output_loc_64(M, N, i, j, bb) : output_loc_32(M, N, i, j, bb);
          const int slot = acc64 ? loc.reg / 2 : loc.reg;
          if (slot == reg && loc.lane == lane) {
            oi = i;
            oj = j;
            ob = bb;
            goto found;
          }
        }
      }
    }
  found:
    if (oi < 0)
      continue;
    Acc acc = c.get(reg);
    for (int k = 0; k < K; ++k) {
      // A is read along dim=M, M*B blocks.
      InputLoc al = input_loc(M, K, B, oi, k, ob, InBits);
      InputLoc bl = input_loc(N, K, B, oj, k, ob, InBits);
      // Pull this lane's worth of the source fragments, then shuffle across.
      auto av_local = ea(al.reg, al.sub); // scalar AElem from this lane
      auto bv_local = eb(bl.reg, bl.sub);
      auto av = detail::wave_shfl(av_local, mfma_a_src_lane(al.lane, cbsz, abid));
      auto bv = detail::wave_shfl(bv_local, mfma_b_src_lane(bl.lane, blgp));
      acc = acc + cast<CElem>(av) * cast<CElem>(bv);
    }
    d.set(reg, acc);
  }
  return d;
}

// detail::e8m0_to_float (OCP-MX E8M0 block-scale -> float) lives in
// fpsan/detail/subbyte_widen.hpp, shared with the gfx1250 WMMA scale path.

// ---- Per-K-block scale lane gather (silicon-verified) ---------------------
// The scaled MFMA carries one E8M0 scale per 32-element K-block, NOT one per
// row/column. For output (row/col `rc`, K-block `kb`) the hardware reads the
// scale from a SPECIFIC lane of the per-lane scale operand:
//   lane = lane_stride*kb + rc   (lane_stride = 16 for 16x16x128's 4 blocks,
//                                 32 for 32x32x64's 2 blocks)
// and `op` (the ScaleAOp/ScaleBOp opsel) selects which of the 4 E8M0 bytes.
// Reverse-engineered + verified full-block-random on MI350. The previous
// uniform-scale model is exactly the special case where every lane carries the
// same scale operand, so existing uniform call sites are unaffected.
//
// `scale` is this lane's copy of the per-lane scale VGPR; ds_bpermute fetches
// the needed lane's copy (byte address = lane*4).
FPSAN_DEVICE inline float scale_block_factor(int scale, int rc, int kb, int lane_stride, int op) {
  const int src = lane_stride * kb + rc;
  const unsigned s = static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(src * 4, scale));
  return e8m0_to_float((s >> (8 * op)) & 0xFFu);
}

// Does the operand fragment type match the f8f6f4 format immediate?
// CBSZ/BLGP: 0=E4M3, 1=E5M2, 2=E2M3, 3=E3M2, 4=E2M1. This checks the 8-bit
// (fp8/bf8) Value<v32_fragment> path; the sub-byte formats fp6/bf6/fp4 (2-4)
// are handled by the separate *_f8f6f4_sub wrappers (raw packed v8i32). (Compares
// the whole fragment type via is_same_v rather than extracting the element with
// vector_element_t, which uses std::declval -- host-only, unusable in
// __device__ context.)
template <class Frag> FPSAN_HOST_DEVICE constexpr bool scaled_fmt_ok(int code) {
  return (code == 0 && std::is_same_v<Frag, v32_fragment<fp8_e4m3>>) ||
         (code == 1 && std::is_same_v<Frag, v32_fragment<fp8_e5m2>>);
}

// ---- Scaled f8f6f4 MFMA software dataflow (16x16x128) ----------------------
// 8-bit operands (E4M3 fp8 or E5M2 bf8), selected per operand by the AFrag /
// BFrag fragment element type. Per-K-block scale: each of the 4 K-blocks
// (32 elements) carries its own E8M0 scale, read from lane 16*kb+row of the
// scale operand (see scale_block_factor). The per-block product is scaled,
// then accumulated -- uniform scale is the special case where all blocks agree.
template <class AFrag, class BFrag, Semantics S, Conversions C>
FPSAN_DEVICE Value<v4f_native, S, C>
mfma_scale_software_16x16x128(Value<AFrag, S, C> a, Value<BFrag, S, C> b, Value<v4f_native, S, C> c,
                              int scale_a, int scale_b, int opA, int opB) {
  using AccS = Value<float, S, C>;
  const int lane = wave_lane_full();
  Value<v4f_native, S, C> d{};
  for (int reg = 0; reg < 4; ++reg) {
    const int i = 4 * (lane / 16) + reg; // output_loc_32 inverse
    const int j = lane % 16;
    AccS acc{};
    for (int kb = 0; kb < 4; ++kb) {
      const AccS sA(scale_block_factor(scale_a, i, kb, 16, opA));
      const AccS sB(scale_block_factor(scale_b, j, kb, 16, opB));
      AccS blk{};
      for (int k = 32 * kb; k < 32 * kb + 32; ++k) {
        const int bs = (k / 64) * 16 + (k % 16);
        const int src = 16 * ((k % 64) / 16);
        auto av = wave_shfl(a.get(bs), src + i);
        auto bv = wave_shfl(b.get(bs), src + j);
        blk = blk + cast<float>(av) * cast<float>(bv);
      }
      acc = acc + blk * sA * sB;
    }
    d.set(reg, c.get(reg) + acc);
  }
  return d;
}

// ---- Scaled f8f6f4 MFMA software dataflow (32x32x64) -----------------------
// Same model as the 16x16x128 helper, with the 32x32x64 fp8 layout (ISA 7.1.5):
//   A[m][k] @ lane 16*(2*((k%32)/16) + m/16) + (m%16), byte (k/32)*16+(k%16);
//   B[k][n] symmetric in n. Output uses output_loc_32(32,32,...), 16 regs/lane.
// Per-K-block scale: 2 blocks of 32, scale read from lane 32*kb+row.
template <class AFrag, class BFrag, Semantics S, Conversions C>
FPSAN_DEVICE Value<v16f_native, S, C>
mfma_scale_software_32x32x64(Value<AFrag, S, C> a, Value<BFrag, S, C> b, Value<v16f_native, S, C> c,
                             int scale_a, int scale_b, int opA, int opB) {
  using AccS = Value<float, S, C>;
  const int lane = wave_lane_full();
  Value<v16f_native, S, C> d{};
  for (int reg = 0; reg < 16; ++reg) {
    int oi = -1, oj = -1;
    for (int i = 0; i < 32 && oi < 0; ++i)
      for (int j = 0; j < 32; ++j) {
        OutputLoc loc = output_loc_32(32, 32, i, j, 0);
        if (loc.reg == reg && loc.lane == lane) {
          oi = i;
          oj = j;
          break;
        }
      }
    if (oi < 0)
      continue;
    AccS acc{};
    for (int kb = 0; kb < 2; ++kb) {
      const AccS sA(scale_block_factor(scale_a, oi, kb, 32, opA));
      const AccS sB(scale_block_factor(scale_b, oj, kb, 32, opB));
      AccS blk{};
      for (int k = 32 * kb; k < 32 * kb + 32; ++k) {
        const int byte = (k / 32) * 16 + (k % 16);
        const int base = 2 * ((k % 32) / 16);
        auto av = wave_shfl(a.get(byte), 16 * (base + oi / 16) + (oi % 16));
        auto bv = wave_shfl(b.get(byte), 16 * (base + oj / 16) + (oj % 16));
        blk = blk + cast<float>(av) * cast<float>(bv);
      }
      acc = acc + blk * sA * sB;
    }
    d.set(reg, c.get(reg) + acc);
  }
  return d;
}

// ---- Sub-byte (fp6 / bf6 / fp4) scaled f8f6f4 MFMA FPSan dataflow ----------
// fp6/bf6 (6-bit) and fp4 (4-bit) operands cannot be Value scalar element types
// (sub-byte types fail the 1+exp+mant == 8*sizeof fp_traits invariant). Instead
// each per-lane operand is the raw packed v8i32 register (the same the builtin
// consumes): in FPSan mode its bits ARE the per-slot finite payload codes,
// packed at the silicon-verified positions -- Width-bit field s occupies
// bits Width*s .. Width*s+Width-1, little-endian across the 8 i32 words.
// This matches the cvt sub-byte convention (no FP4/FP6 Value<> element
// type; widen uses the canonical subbyte cast policy). Verified on MI350 by
// a full-block-random match of the builtin for every same-width A/B combo.
//
// Cross-lane gather uses ds_bpermute on the individual 32-bit words (the field
// may straddle two words for fp6). Mixed-width A/B (e.g. fp8 x fp4) is NOT
// supported: the two formats use different k->slot orderings on hardware (the
// permutations cancel in a same-format test but not when mixed), so the
// wrappers static_assert that A and B share a bit width.

// detail::subbyte_widen_to comes from fpsan/detail/subbyte_widen.hpp,
// shared with amdgcn_cvt.hpp.

// Gather the Width-bit field at slot s from `srclane`'s packed register `pw`
// (8 i32, this lane's copy) and widen it to an f32 payload.
template <int Width, Semantics S, Conversions C>
FPSAN_DEVICE Value<float, S, C> sub_gather_widen(const int (&pw)[8], int s, int srclane) {
  const int p = Width * s, wi = p >> 5, off = p & 31;
  unsigned w0 = static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(srclane * 4, pw[wi]));
  unsigned field = w0 >> off;
  if (off > 32 - Width) {
    unsigned w1 = static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(srclane * 4, pw[wi + 1]));
    field |= w1 << (32 - off);
  }
  field &= (1u << Width) - 1u;
  return subbyte_widen_to<float, Width, S, C>(field);
}

// Sub-byte x sub-byte. A and B may have DIFFERENT widths (e.g. fp6 x fp4):
// the hardware pairs A field-index s with B field-index s regardless of width
// (silicon-verified -- the per-width physical->k order is identical across all
// sub formats, so equal field indices pair the same k). Per-K-block scale via
// scale_block_factor.
template <int WA, int WB, Semantics S, Conversions C>
FPSAN_DEVICE Value<v4f_native, S, C>
mfma_scale_sub_16x16x128(const int (&aw)[8], const int (&bw)[8], Value<v4f_native, S, C> c,
                         int scale_a, int scale_b, int opA, int opB) {
  using AccS = Value<float, S, C>;
  const int lane = wave_lane_full();
  Value<v4f_native, S, C> d{};
  for (int reg = 0; reg < 4; ++reg) {
    const int i = 4 * (lane / 16) + reg;
    const int j = lane % 16;
    AccS acc{};
    for (int kb = 0; kb < 4; ++kb) {
      const AccS sA(scale_block_factor(scale_a, i, kb, 16, opA));
      const AccS sB(scale_block_factor(scale_b, j, kb, 16, opB));
      AccS blk{};
      for (int k = 32 * kb; k < 32 * kb + 32; ++k) {
        const int s = (k / 64) * 16 + (k % 16);
        const int src = 16 * ((k % 64) / 16);
        auto av = sub_gather_widen<WA, S, C>(aw, s, src + i);
        auto bv = sub_gather_widen<WB, S, C>(bw, s, src + j);
        blk = blk + av * bv;
      }
      acc = acc + blk * sA * sB;
    }
    d.set(reg, c.get(reg) + acc);
  }
  return d;
}

template <int WA, int WB, Semantics S, Conversions C>
FPSAN_DEVICE Value<v16f_native, S, C>
mfma_scale_sub_32x32x64(const int (&aw)[8], const int (&bw)[8], Value<v16f_native, S, C> c,
                        int scale_a, int scale_b, int opA, int opB) {
  using AccS = Value<float, S, C>;
  const int lane = wave_lane_full();
  Value<v16f_native, S, C> d{};
  for (int reg = 0; reg < 16; ++reg) {
    int oi = -1, oj = -1;
    for (int i = 0; i < 32 && oi < 0; ++i)
      for (int j = 0; j < 32; ++j) {
        OutputLoc loc = output_loc_32(32, 32, i, j, 0);
        if (loc.reg == reg && loc.lane == lane) {
          oi = i;
          oj = j;
          break;
        }
      }
    if (oi < 0)
      continue;
    AccS acc{};
    for (int kb = 0; kb < 2; ++kb) {
      const AccS sA(scale_block_factor(scale_a, oi, kb, 32, opA));
      const AccS sB(scale_block_factor(scale_b, oj, kb, 32, opB));
      AccS blk{};
      for (int k = 32 * kb; k < 32 * kb + 32; ++k) {
        const int s = (k / 32) * 16 + (k % 16);
        const int base = 2 * ((k % 32) / 16);
        const int alane = 16 * (base + oi / 16) + (oi % 16);
        const int blane = 16 * (base + oj / 16) + (oj % 16);
        auto av = sub_gather_widen<WA, S, C>(aw, s, alane);
        auto bv = sub_gather_widen<WB, S, C>(bw, s, blane);
        blk = blk + av * bv;
      }
      acc = acc + blk * sA * sB;
    }
    d.set(reg, c.get(reg) + acc);
  }
  return d;
}

// ---- Mixed-WIDTH 8-bit x sub-byte scaled f8f6f4 dataflow -------------------
// fp8/bf8 (A) x fp6/bf6/fp4 (B) or the mirror. The two formats use DIFFERENT
// physical k orderings, so equal field indices do NOT pair the same k. The
// 8-bit operand uses its native fp8 model; the sub-byte operand uses the
// silicon-RE'd "mix model" that pairs it correctly with fp8 (reverse-engineered
// and proven full-block-random on MI350 for both shapes):
//   16x16x128 sub mix slot: lane = 16*q0 + idx, field p0, with
//     q0 = 2*((k>>6)&1) + ((k>>5)&1),  p0 = 16*((k>>4)&1) + (k&15)
//   32x32x64  sub mix slot: lane = 16*(2*(k/32) + idx/16) + (idx%16),
//     field p0 = 16*((k%32)/16) + (k%16)
// The 8-bit operand stays a Value<v32_fragment> (proper fp8 payload casting);
// the sub operand is the raw packed v8i32 (bits are finite payload codes).
// `aIsSub` picks which side is sub. Per-K-block scale via scale_block_factor.
template <bool AIsSub, int Wsub, class Fp8Frag, Semantics S, Conversions C>
FPSAN_DEVICE Value<v4f_native, S, C>
mfma_scale_mixed_16x16x128(Value<Fp8Frag, S, C> fp8, const int (&sub)[8], Value<v4f_native, S, C> c,
                           int scale_a, int scale_b, int opA, int opB) {
  using AccS = Value<float, S, C>;
  const int lane = wave_lane_full();
  Value<v4f_native, S, C> d{};
  for (int reg = 0; reg < 4; ++reg) {
    const int i = 4 * (lane / 16) + reg;
    const int j = lane % 16;
    AccS acc{};
    for (int kb = 0; kb < 4; ++kb) {
      const AccS sA(scale_block_factor(scale_a, i, kb, 16, opA));
      const AccS sB(scale_block_factor(scale_b, j, kb, 16, opB));
      AccS blk{};
      for (int k = 32 * kb; k < 32 * kb + 32; ++k) {
        const int bs = (k / 64) * 16 + (k % 16);            // fp8 byte slot
        const int fsrc = 16 * ((k % 64) / 16);              // fp8 lane group base
        const int q0 = 2 * ((k >> 6) & 1) + ((k >> 5) & 1); // sub mix lane grp
        const int p0 = 16 * ((k >> 4) & 1) + (k & 15);      // sub mix field
        const int idx = AIsSub ? i : j;                     // sub operand uses this output index
        const int slane = 16 * q0 + idx;
        const int flane = fsrc + (AIsSub ? j : i);
        auto fv = cast<float>(wave_shfl(fp8.get(bs), flane));
        auto sv = sub_gather_widen<Wsub, S, C>(sub, p0, slane);
        blk = blk + (AIsSub ? sv * fv : fv * sv);
      }
      acc = acc + blk * sA * sB;
    }
    d.set(reg, c.get(reg) + acc);
  }
  return d;
}

template <bool AIsSub, int Wsub, class Fp8Frag, Semantics S, Conversions C>
FPSAN_DEVICE Value<v16f_native, S, C>
mfma_scale_mixed_32x32x64(Value<Fp8Frag, S, C> fp8, const int (&sub)[8], Value<v16f_native, S, C> c,
                          int scale_a, int scale_b, int opA, int opB) {
  using AccS = Value<float, S, C>;
  const int lane = wave_lane_full();
  Value<v16f_native, S, C> d{};
  for (int reg = 0; reg < 16; ++reg) {
    int oi = -1, oj = -1;
    for (int i = 0; i < 32 && oi < 0; ++i)
      for (int j = 0; j < 32; ++j) {
        OutputLoc loc = output_loc_32(32, 32, i, j, 0);
        if (loc.reg == reg && loc.lane == lane) {
          oi = i;
          oj = j;
          break;
        }
      }
    if (oi < 0)
      continue;
    AccS acc{};
    for (int kb = 0; kb < 2; ++kb) {
      const AccS sA(scale_block_factor(scale_a, oi, kb, 32, opA));
      const AccS sB(scale_block_factor(scale_b, oj, kb, 32, opB));
      AccS blk{};
      const int fidx = AIsSub ? oj : oi; // fp8 output index
      const int sidx = AIsSub ? oi : oj; // sub output index
      for (int k = 32 * kb; k < 32 * kb + 32; ++k) {
        const int byte = (k / 32) * 16 + (k % 16);
        const int base = 2 * ((k % 32) / 16);
        const int flane = 16 * (base + fidx / 16) + (fidx % 16);
        const int sbase = 2 * (k / 32);
        const int slane = 16 * (sbase + sidx / 16) + (sidx % 16);
        const int p0 = 16 * ((k % 32) / 16) + (k % 16);
        auto fv = cast<float>(wave_shfl(fp8.get(byte), flane));
        auto sv = sub_gather_widen<Wsub, S, C>(sub, p0, slane);
        blk = blk + (AIsSub ? sv * fv : fv * sv);
      }
      acc = acc + blk * sA * sB;
    }
    d.set(reg, c.get(reg) + acc);
  }
  return d;
}

// detail::f8f6f4_width (element bit width of the format immediate) lives in
// fpsan/detail/subbyte_widen.hpp, shared with the gfx1250 WMMA f8f6f4 path.

} // namespace detail

// =============================================================================
// CDNA MFMA wrappers.
//
// Each wrapper:
//   - Native mode: bit-casts the Value fragments to the native vector ABI and
//     calls the matching __builtin_amdgcn_mfma_*.
//   - FPSan mode: runs the software MFMA in detail::mfma_software, gathering
//     A and B elements via wave_shfl and accumulating in the payload ring.
//
// The cbsz/abid/blgp immediates (A-broadcast block size/id and B lane-group
// permutation) default to 0 (identity) in the wrappers below.  Templated
// overloads taking explicit non-zero values can be added once a customer
// needs them.
//
// Gated by __has_builtin so the header still compiles on archs that don't
// expose MFMA at all; the host-pass exemption keeps the wrappers visible to
// HIP test source parsing even when device-pass is hidden.
// =============================================================================

// Helper: same-shape extractor for a vector-of-16-bit-fragment.
// input_loc() returns (reg, sub) where `reg` is the 32-bit dword slot and
// `sub` is the sub-element within that dword (for 16-bit operands, 2 elements
// per dword). The element's index inside the per-lane fragment is therefore
// 2*reg + sub -- ignoring `sub` would read only the low half of every dword
// and silently drop half of the K elements.
#define FPSAN_MFMA_EXTRACT_VEC16(Frag) [&](int reg, int sub) { return Frag.get(2 * reg + sub); }
#define FPSAN_MFMA_EXTRACT_VEC8(Frag) FPSAN_MFMA_EXTRACT_VEC16(Frag)

// Helper: 8-bit-packed extractor for a long-int (Wi) fragment containing 8
// fp8/bf8 bytes per lane.  Returns a Value<FP8> built from the chosen byte.
#define FPSAN_MFMA_EXTRACT_FP8(Frag, FP8Type, Sem, Conv)                                           \
  [&](int reg, int sub) {                                                                          \
    const std::uint64_t u = static_cast<std::uint64_t>(Frag.to_storage_bits());                    \
    const int bit_off = reg * 32 + sub * 8;                                                        \
    const std::uint8_t byte = static_cast<std::uint8_t>((u >> bit_off) & 0xFFu);                   \
    if constexpr (Sem == Semantics::Native)                                                        \
      return Value<FP8Type, Sem, Conv>(FP8Type(byte));                                             \
    else                                                                                           \
      return Value<FP8Type, Sem, Conv>::from_fpsan_payload(byte);                                  \
  }

// ---- F16 / BF16 inputs, F32 accumulator (gfx950-insts) ----------------------
#define FPSAN_DEFINE_MFMA_F32_VEC8(NAME, M_, N_, K_, AVec_, AElem_, BUILTIN)                       \
  template <int CBSZ = 0, int ABID = 0, int BLGP = 0, Semantics S = Semantics::Native,             \
            Conversions Cv = Conversions::Explicit>                                                \
  FPSAN_DEVICE Value<v4f_native, S, Cv> NAME(Value<AVec_, S, Cv> a, Value<AVec_, S, Cv> b,         \
                                             Value<v4f_native, S, Cv> c) {                         \
    if constexpr (M_ == 32 && N_ == 32) {                                                          \
      static_assert(M_ != 32, "use the 32x32x16 wrapper for v16f acc");                            \
    }                                                                                              \
    if constexpr (S == Semantics::Native) {                                                        \
      auto d = BUILTIN(a.to_float(), b.to_float(), c.to_float(), CBSZ, ABID, BLGP);                \
      return Value<v4f_native, S, Cv>(d);                                                          \
    } else {                                                                                       \
      return detail::mfma_software<M_, N_, K_, 1, /*InBits=*/16, float, S, Cv>(                    \
          a, b, c, FPSAN_MFMA_EXTRACT_VEC8(a), FPSAN_MFMA_EXTRACT_VEC8(b), CBSZ, ABID, BLGP);      \
    }                                                                                              \
  }

#define FPSAN_DEFINE_MFMA_F32_VEC8_BIG(NAME, M_, N_, K_, AVec_, BUILTIN)                           \
  template <int CBSZ = 0, int ABID = 0, int BLGP = 0, Semantics S = Semantics::Native,             \
            Conversions Cv = Conversions::Explicit>                                                \
  FPSAN_DEVICE Value<v16f_native, S, Cv> NAME(Value<AVec_, S, Cv> a, Value<AVec_, S, Cv> b,        \
                                              Value<v16f_native, S, Cv> c) {                       \
    if constexpr (S == Semantics::Native) {                                                        \
      auto d = BUILTIN(a.to_float(), b.to_float(), c.to_float(), CBSZ, ABID, BLGP);                \
      return Value<v16f_native, S, Cv>(d);                                                         \
    } else {                                                                                       \
      return detail::mfma_software<M_, N_, K_, 1, /*InBits=*/16, float, S, Cv>(                    \
          a, b, c, FPSAN_MFMA_EXTRACT_VEC8(a), FPSAN_MFMA_EXTRACT_VEC8(b), CBSZ, ABID, BLGP);      \
    }                                                                                              \
  }

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_f16)
FPSAN_DEFINE_MFMA_F32_VEC8(amdgcn_mfma_f32_16x16x32_f16, 16, 16, 32, v8h_native, _Float16,
                           __builtin_amdgcn_mfma_f32_16x16x32_f16)
FPSAN_DEFINE_MFMA_F32_VEC8(amdgcn_mfma_f32_16x16x32_bf16, 16, 16, 32, v8bf_native, __bf16,
                           __builtin_amdgcn_mfma_f32_16x16x32_bf16)
FPSAN_DEFINE_MFMA_F32_VEC8_BIG(amdgcn_mfma_f32_32x32x16_f16, 32, 32, 16, v8h_native,
                               __builtin_amdgcn_mfma_f32_32x32x16_f16)
FPSAN_DEFINE_MFMA_F32_VEC8_BIG(amdgcn_mfma_f32_32x32x16_bf16, 32, 32, 16, v8bf_native,
                               __builtin_amdgcn_mfma_f32_32x32x16_bf16)
#endif

#undef FPSAN_DEFINE_MFMA_F32_VEC8
#undef FPSAN_DEFINE_MFMA_F32_VEC8_BIG

// ---- Legacy gfx9 f16 / bf16_1k MFMA shapes (still present on gfx950) --------
// Small-K / multi-block dense MFMAs whose A,B per-lane fragment is 4 elements
// of f16 (or bf16, the "_1k" variants). They reuse the dense input_loc /
// output_loc_32 dataflow with InBits=16 and a block count B_. The extractor
// maps (reg, sub) -> element 2*reg+sub (2 16-bit elements per dword), the same
// as the CDNA4 vec8 path. All ten layouts + signatures are silicon-verified on
// MI350 (full-block random match of the builtin vs the input_loc/output_loc_32
// reference); pinned by the LegacyMfmaF16_* golden tests.
//
// Note: Clang exposes non-1k bf16 shapes (mfma_f32_*bf16 with a 2-element
// fragment) via __has_builtin on gfx942, but the backend cannot select them for
// this target. They are intentionally omitted rather than exposing wrappers
// that fail to lower.
#define FPSAN_DEFINE_MFMA_F32_VEC4(NAME, M_, N_, K_, B_, AVec_, CVec_, BUILTIN)                    \
  template <int CBSZ = 0, int ABID = 0, int BLGP = 0, Semantics S = Semantics::Native,             \
            Conversions Cv = Conversions::Explicit>                                                \
  FPSAN_DEVICE Value<CVec_, S, Cv> NAME(Value<AVec_, S, Cv> a, Value<AVec_, S, Cv> b,              \
                                        Value<CVec_, S, Cv> c) {                                   \
    if constexpr (S == Semantics::Native) {                                                        \
      auto d = BUILTIN(a.to_float(), b.to_float(), c.to_float(), CBSZ, ABID, BLGP);                \
      return Value<CVec_, S, Cv>(d);                                                               \
    } else {                                                                                       \
      return detail::mfma_software<M_, N_, K_, B_, /*InBits=*/16, float, S, Cv>(                   \
          a, b, c, FPSAN_MFMA_EXTRACT_VEC8(a), FPSAN_MFMA_EXTRACT_VEC8(b), CBSZ, ABID, BLGP);      \
    }                                                                                              \
  }

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16f16)
FPSAN_DEFINE_MFMA_F32_VEC4(amdgcn_mfma_f32_16x16x16f16, 16, 16, 16, 1, v4h_native, v4f_native,
                           __builtin_amdgcn_mfma_f32_16x16x16f16)
FPSAN_DEFINE_MFMA_F32_VEC4(amdgcn_mfma_f32_16x16x4f16, 16, 16, 4, 4, v4h_native, v16f_native,
                           __builtin_amdgcn_mfma_f32_16x16x4f16)
FPSAN_DEFINE_MFMA_F32_VEC4(amdgcn_mfma_f32_32x32x8f16, 32, 32, 8, 1, v4h_native, v16f_native,
                           __builtin_amdgcn_mfma_f32_32x32x8f16)
FPSAN_DEFINE_MFMA_F32_VEC4(amdgcn_mfma_f32_32x32x4f16, 32, 32, 4, 2, v4h_native, v32f_native,
                           __builtin_amdgcn_mfma_f32_32x32x4f16)
FPSAN_DEFINE_MFMA_F32_VEC4(amdgcn_mfma_f32_4x4x4f16, 4, 4, 4, 16, v4h_native, v4f_native,
                           __builtin_amdgcn_mfma_f32_4x4x4f16)
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
FPSAN_DEFINE_MFMA_F32_VEC4(amdgcn_mfma_f32_16x16x16bf16_1k, 16, 16, 16, 1, v4bf_native, v4f_native,
                           __builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
FPSAN_DEFINE_MFMA_F32_VEC4(amdgcn_mfma_f32_16x16x4bf16_1k, 16, 16, 4, 4, v4bf_native, v16f_native,
                           __builtin_amdgcn_mfma_f32_16x16x4bf16_1k)
FPSAN_DEFINE_MFMA_F32_VEC4(amdgcn_mfma_f32_32x32x8bf16_1k, 32, 32, 8, 1, v4bf_native, v16f_native,
                           __builtin_amdgcn_mfma_f32_32x32x8bf16_1k)
FPSAN_DEFINE_MFMA_F32_VEC4(amdgcn_mfma_f32_32x32x4bf16_1k, 32, 32, 4, 2, v4bf_native, v32f_native,
                           __builtin_amdgcn_mfma_f32_32x32x4bf16_1k)
FPSAN_DEFINE_MFMA_F32_VEC4(amdgcn_mfma_f32_4x4x4bf16_1k, 4, 4, 4, 16, v4bf_native, v4f_native,
                           __builtin_amdgcn_mfma_f32_4x4x4bf16_1k)
#endif

#undef FPSAN_DEFINE_MFMA_F32_VEC4

// ---- FP8 / BF8 inputs, F32 accumulator (fp8-insts) --------------------------
// A and B are each a v8 fragment of fp8/bf8 = 8 bytes per lane.  The LLVM ABI
// takes them as 'long' (Wi). CDNA3/gfx94x uses AMD FNUZ FP8/BF8 encodings
// (amd_fp8_e4m3/e5m2); RDNA4/CDNA4 use the existing OCP encodings
// (fp8_e4m3/e5m2). The wrapper names are the same, but the fragment types are
// architecture-specific to prevent silently interpreting one byte format as the
// other.
#define FPSAN_DEFINE_MFMA_F32_FP8(NAME, M_, N_, K_, AVec_, BVec_, CFragVec_, BUILTIN)              \
  template <int CBSZ = 0, int ABID = 0, int BLGP = 0, Semantics S = Semantics::Native,             \
            Conversions Cv = Conversions::Explicit>                                                \
  FPSAN_DEVICE Value<CFragVec_, S, Cv> NAME(Value<AVec_, S, Cv> a, Value<BVec_, S, Cv> b,          \
                                            Value<CFragVec_, S, Cv> c) {                           \
    if constexpr (S == Semantics::Native) {                                                        \
      const long ai = __builtin_bit_cast(long, a.to_float());                                      \
      const long bi = __builtin_bit_cast(long, b.to_float());                                      \
      auto d = BUILTIN(ai, bi, c.to_float(), CBSZ, ABID, BLGP);                                    \
      return Value<CFragVec_, S, Cv>(d);                                                           \
    } else {                                                                                       \
      /* 8-bit operands: 4 bytes per dword, element index = 4*reg + sub. */                        \
      auto ea = [&](int reg, int sub) { return a.get(4 * reg + sub); };                            \
      auto eb = [&](int reg, int sub) { return b.get(4 * reg + sub); };                            \
      return detail::mfma_software<M_, N_, K_, 1, /*InBits=*/8, float, S, Cv>(a, b, c, ea, eb,     \
                                                                              CBSZ, ABID, BLGP);   \
    }                                                                                              \
  }

#if !defined(__HIP_DEVICE_COMPILE__) ||                                                            \
    (__has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8) && !defined(__gfx940__) &&          \
     !defined(__gfx941__) && !defined(__gfx942__))
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_16x16x32_fp8_fp8, 16, 16, 32, v8e4m3_native,
                          v8e4m3_native, v4f_native, __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_16x16x32_fp8_bf8, 16, 16, 32, v8e4m3_native,
                          v8e5m2_native, v4f_native, __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_16x16x32_bf8_fp8, 16, 16, 32, v8e5m2_native,
                          v8e4m3_native, v4f_native, __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_16x16x32_bf8_bf8, 16, 16, 32, v8e5m2_native,
                          v8e5m2_native, v4f_native, __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_32x32x16_fp8_fp8, 32, 32, 16, v8e4m3_native,
                          v8e4m3_native, v16f_native, __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_32x32x16_fp8_bf8, 32, 32, 16, v8e4m3_native,
                          v8e5m2_native, v16f_native, __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_32x32x16_bf8_fp8, 32, 32, 16, v8e5m2_native,
                          v8e4m3_native, v16f_native, __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_32x32x16_bf8_bf8, 32, 32, 16, v8e5m2_native,
                          v8e5m2_native, v16f_native, __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8)
#endif

#if !defined(__HIP_DEVICE_COMPILE__) ||                                                            \
    (__has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8) &&                                  \
     (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)))
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_16x16x32_fp8_fp8, 16, 16, 32, v8amd_e4m3_native,
                          v8amd_e4m3_native, v4f_native, __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_16x16x32_fp8_bf8, 16, 16, 32, v8amd_e4m3_native,
                          v8amd_e5m2_native, v4f_native, __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_16x16x32_bf8_fp8, 16, 16, 32, v8amd_e5m2_native,
                          v8amd_e4m3_native, v4f_native, __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_16x16x32_bf8_bf8, 16, 16, 32, v8amd_e5m2_native,
                          v8amd_e5m2_native, v4f_native, __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_32x32x16_fp8_fp8, 32, 32, 16, v8amd_e4m3_native,
                          v8amd_e4m3_native, v16f_native,
                          __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_32x32x16_fp8_bf8, 32, 32, 16, v8amd_e4m3_native,
                          v8amd_e5m2_native, v16f_native,
                          __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_32x32x16_bf8_fp8, 32, 32, 16, v8amd_e5m2_native,
                          v8amd_e4m3_native, v16f_native,
                          __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8)
FPSAN_DEFINE_MFMA_F32_FP8(amdgcn_mfma_f32_32x32x16_bf8_bf8, 32, 32, 16, v8amd_e5m2_native,
                          v8amd_e5m2_native, v16f_native,
                          __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8)
#endif

#undef FPSAN_DEFINE_MFMA_F32_FP8

// ---- XF32 inputs, F32 accumulator (cdna3 xf32-insts) -----------------------
// xF32 takes FP32 storage in a v2f per-lane fragment, rounds A/B mantissas to
// 10 bits internally, and accumulates/output in FP32. FPSan mode models the
// algebraic dataflow over f32 payloads; layout tests use exact small values so
// the reduced-precision hardware path equals the scalar reference.
#define FPSAN_DEFINE_MFMA_F32_XF32(NAME, M_, N_, K_, CVec_, BUILTIN)                               \
  template <int CBSZ = 0, int ABID = 0, int BLGP = 0, Semantics S = Semantics::Native,             \
            Conversions Cv = Conversions::Explicit>                                                \
  FPSAN_DEVICE Value<CVec_, S, Cv> NAME(Value<v2f_native, S, Cv> a, Value<v2f_native, S, Cv> b,    \
                                        Value<CVec_, S, Cv> c) {                                   \
    if constexpr (S == Semantics::Native) {                                                        \
      auto d = BUILTIN(a.to_float(), b.to_float(), c.to_float(), CBSZ, ABID, BLGP);                \
      return Value<CVec_, S, Cv>(d);                                                               \
    } else {                                                                                       \
      return detail::mfma_software<M_, N_, K_, 1, /*InBits=*/32, float, S, Cv>(                    \
          a, b, c, [&](int reg, int /*sub*/) { return a.get(reg); },                               \
          [&](int reg, int /*sub*/) { return b.get(reg); }, CBSZ, ABID, BLGP);                     \
    }                                                                                              \
  }

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_mfma_f32_16x16x8_xf32)
FPSAN_DEFINE_MFMA_F32_XF32(amdgcn_mfma_f32_16x16x8_xf32, 16, 16, 8, v4f_native,
                           __builtin_amdgcn_mfma_f32_16x16x8_xf32)
FPSAN_DEFINE_MFMA_F32_XF32(amdgcn_mfma_f32_32x32x4_xf32, 32, 32, 4, v16f_native,
                           __builtin_amdgcn_mfma_f32_32x32x4_xf32)
#endif

#undef FPSAN_DEFINE_MFMA_F32_XF32

// ---- F64 input + F64 accumulator (mai-insts; available on CDNA3+) ----------
// f64_16x16x4: 16x16x4 with scalar f64 inputs + v4d accumulator.
// f64_4x4x4:   4x4x4 with scalar f64 inputs + scalar f64 accumulator.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_mfma_f64_16x16x4f64)
template <int CBSZ = 0, int ABID = 0, int BLGP = 0, Semantics S = Semantics::Native,
          Conversions Cv = Conversions::Explicit>
FPSAN_DEVICE Value<v4d_native, S, Cv> amdgcn_mfma_f64_16x16x4f64(Value<double, S, Cv> a,
                                                                 Value<double, S, Cv> b,
                                                                 Value<v4d_native, S, Cv> c) {
  static_assert(CBSZ == 0 && ABID == 0,
                "f64 MFMA does not support CBSZ/ABID broadcast controls; use the third "
                "immediate for NEG");
  if constexpr (S == Semantics::Native) {
    auto d = __builtin_amdgcn_mfma_f64_16x16x4f64(a.to_float(), b.to_float(), c.to_float(), CBSZ,
                                                  ABID, BLGP);
    return Value<v4d_native, S, Cv>(d);
  } else {
    // Scalar-A/B variant: each lane carries one f64 element of A and B.
    // The dataflow gathers across lanes per (i, k) / (j, k); the
    // "register-0" path of the extractor is the lane's scalar Value.
    // F64 MFMA uses the third immediate as NEG[2:0], not BLGP: bit 0
    // negates A, bit 1 negates B, and bit 2 negates C.
    auto an = a;
    auto bn = b;
    auto cn = c;
    if constexpr ((BLGP & 1) != 0)
      an = -an;
    if constexpr ((BLGP & 2) != 0)
      bn = -bn;
    if constexpr ((BLGP & 4) != 0)
      cn = -cn;
    auto ea = [&](int /*reg*/, int /*sub*/) { return an; };
    auto eb = [&](int /*reg*/, int /*sub*/) { return bn; };
    return detail::mfma_software<16, 16, 4, 1, /*InBits=*/64, double, S, Cv>(an, bn, cn, ea, eb, 0,
                                                                             0, 0);
  }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_mfma_f64_4x4x4f64)
template <int CBSZ = 0, int ABID = 0, int BLGP = 0, Semantics S = Semantics::Native,
          Conversions Cv = Conversions::Explicit>
FPSAN_DEVICE Value<double, S, Cv>
amdgcn_mfma_f64_4x4x4f64(Value<double, S, Cv> a, Value<double, S, Cv> b, Value<double, S, Cv> c) {
  static_assert(CBSZ == 0 && ABID == 0,
                "f64 MFMA does not support CBSZ/ABID broadcast controls; use the third "
                "immediate for NEG");
  if constexpr (S == Semantics::Native) {
    auto d = __builtin_amdgcn_mfma_f64_4x4x4f64(a.to_float(), b.to_float(), c.to_float(), CBSZ,
                                                ABID, BLGP);
    return Value<double, S, Cv>(d);
  } else {
    // V_MFMA_F64_4X4X4_4B_F64: 4 independent 4x4 blocks (B=4), each a K=4
    // contraction, one scalar output per lane. The per-lane layout was
    // reverse-engineered on real MI350 (the dense input_loc/output_loc
    // formulas do NOT apply to this shape -- K=4 < 16 lanes/block, so it is
    // pinned directly to silicon):
    //   output lane L holds D[i][j] of block b with
    //       i = L / 16,  b = (L % 16) / 4,  j = L % 4
    //   A[i][k] of block b lives at lane 16*k + 4*b + i
    //   B[k][j] of block b lives at lane 16*k + 4*b + j
    //   D[L] = C[L] + sum_{k=0..3} A[..]*B[..]
    // Pinned by MfmaF64_4x4x4.LayoutMatchesHardware.
    const int lane = detail::wave_lane_full();
    const int i = lane / 16, blk = (lane % 16) / 4, j = lane % 4;
    auto an = a;
    auto bn = b;
    Value<double, S, Cv> acc = c;
    if constexpr ((BLGP & 1) != 0)
      an = -an;
    if constexpr ((BLGP & 2) != 0)
      bn = -bn;
    if constexpr ((BLGP & 4) != 0)
      acc = -acc;
    for (int k = 0; k < 4; ++k) {
      auto av = detail::wave_shfl(an, 16 * k + 4 * blk + i);
      auto bv = detail::wave_shfl(bn, 16 * k + 4 * blk + j);
      acc = acc + av * bv;
    }
    return acc;
  }
}
#endif

// ---- Legacy gfx9 f32-input MFMA shapes (still present on gfx950) -----------
// Small-K / multi-block dense MFMAs with SCALAR f32 A and B per lane (one f32
// element each; the K elements of a row/column are spread across lanes, and the
// `B` blocks are independent MxN products). Float forwards to the builtin;
// FPSan reuses the dense detail::mfma_software dataflow, which already handles
// the block dimension via input_loc/output_loc_32. All five shapes' layouts +
// signatures are silicon-verified on MI350 (full-block random match of the
// builtin against the input_loc/output_loc_32 reference):
//   16x16x4f32 (B=1) 16x16x1f32 (B=4) 32x32x2f32 (B=1) 32x32x1f32 (B=2)
//   4x4x1f32   (B=16)
// Pinned by the LegacyMfmaF32_* golden tests (Layout + FPSan).
#define FPSAN_DEFINE_MFMA_F32_SCALAR(NAME, M_, N_, K_, B_, CVEC_, BUILTIN)                         \
  template <int CBSZ = 0, int ABID = 0, int BLGP = 0, Semantics S = Semantics::Native,             \
            Conversions Cv = Conversions::Explicit>                                                \
  FPSAN_DEVICE Value<CVEC_, S, Cv> NAME(Value<float, S, Cv> a, Value<float, S, Cv> b,              \
                                        Value<CVEC_, S, Cv> c) {                                   \
    if constexpr (S == Semantics::Native) {                                                        \
      auto d = BUILTIN(a.to_float(), b.to_float(), c.to_float(), CBSZ, ABID, BLGP);                \
      return Value<CVEC_, S, Cv>(d);                                                               \
    } else {                                                                                       \
      auto ea = [&](int, int) { return a; };                                                       \
      auto eb = [&](int, int) { return b; };                                                       \
      return detail::mfma_software<M_, N_, K_, B_, /*InBits=*/32, float, S, Cv>(a, b, c, ea, eb,   \
                                                                                CBSZ, ABID, BLGP); \
    }                                                                                              \
  }

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_mfma_f32_16x16x4f32)
FPSAN_DEFINE_MFMA_F32_SCALAR(amdgcn_mfma_f32_16x16x4f32, 16, 16, 4, 1, v4f_native,
                             __builtin_amdgcn_mfma_f32_16x16x4f32)
FPSAN_DEFINE_MFMA_F32_SCALAR(amdgcn_mfma_f32_16x16x1f32, 16, 16, 1, 4, v16f_native,
                             __builtin_amdgcn_mfma_f32_16x16x1f32)
FPSAN_DEFINE_MFMA_F32_SCALAR(amdgcn_mfma_f32_32x32x2f32, 32, 32, 2, 1, v16f_native,
                             __builtin_amdgcn_mfma_f32_32x32x2f32)
FPSAN_DEFINE_MFMA_F32_SCALAR(amdgcn_mfma_f32_32x32x1f32, 32, 32, 1, 2, v32f_native,
                             __builtin_amdgcn_mfma_f32_32x32x1f32)
FPSAN_DEFINE_MFMA_F32_SCALAR(amdgcn_mfma_f32_4x4x1f32, 4, 4, 1, 16, v4f_native,
                             __builtin_amdgcn_mfma_f32_4x4x1f32)
#endif

#undef FPSAN_DEFINE_MFMA_F32_SCALAR

// ---- Scaled MFMA (mfma_scale_f32_*_f8f6f4): the new gfx950 family ----------
// 16x16x128 / 32x32x64 mixed-precision MFMA with per-K-block scales.
//
// A and B are each 32 bytes per lane, handed to the builtin as a v8i32. The
// per-element interpretation (fp8 / bf8 / fp6 / bf6 / fp4) is selected by the
// CBSZ (A) and BLGP (B) format immediates. The `scale_a` / `scale_b` runtime
// args carry the OCP-MX E8M0 block scales; the `ScaleAOp` / `ScaleBOp` opsel
// immediates select which scale byte applies.
//
// The exact 9-argument builtin signature on gfx950 is
//   (A:v8i32, B:v8i32, C, cbsz_imm, blgp_imm, opselA_imm, scaleA_rt,
//    opselB_imm, scaleB_rt)
// -- note there is NO ABID operand (unlike the dense MFMAs), and A/B must be
// bit-cast to v8i32. The template keeps ABID as an unused parameter only for
// call-site stability with the dense wrappers.
//
// Status: Float path forwards to the real builtin; FPSan path implements the
// OCP-MX dataflow below. Both exercised on MI350. These wrappers cover the
// 8-bit fp8/bf8 operands (Value<v32_fragment>); the sub-byte fp6/bf6/fp4
// formats are covered by the *_f8f6f4_sub wrappers further down (raw packed
// v8i32, same lane map with Width-bit fields), also silicon-verified.
//
// Layout / scale, reverse-engineered + cross-checked against the CDNA4 ISA on
// real MI350 (default modifiers; CBSZ=BLGP=0 => A,B are E4M3 fp8):
//   * K=128 fp8 per-lane A fragment (32 bytes): with g = lane/16,
//     A[i][k] lives at lane 16*((k%64)/16) + i, byte (k/64)*16 + (k%16);
//     B[k][j] symmetric in j. (This is NOT the dense input_loc layout.)
//   * Output D[i][j] uses output_loc_32(16,16,...): reg = i%4, lane=(i/4)*16+j.
//   * Scale is E8M0 (2^(byte-127), 0xFF=NaN), one per 32-element K-block. Each
//     K-block kb reads its scale from a SPECIFIC lane of the scale operand:
//     lane = 16*kb + row (16x16x128) / 32*kb + row (32x32x64), byte = opsel.
//     (Silicon-RE'd: detail::scale_block_factor.)
//
// FPSan model: D[i][j] = C[i][j] + sum_kb 2^scaleA(i,kb) * 2^scaleB(j,kb) *
// (sum_{k in block kb} A[i][k]*B[k][j]) in the payload ring. This is correct
// for arbitrary PER-K-BLOCK scales (non-uniform across the row/column); uniform
// scale is the special case where every lane holds the same scale operand.
// Verified full-block-random on MI350 for all opsels.
#if !defined(__HIP_DEVICE_COMPILE__) ||                                                            \
    __has_builtin(__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4)
template <int CBSZ = 0, int ABID = 0, int BLGP = 0, int ScaleAOp = 0, int ScaleBOp = 0,
          Semantics S = Semantics::Native, Conversions Cv = Conversions::Explicit,
          class AFrag = v32e4m3_native, class BFrag = v32e4m3_native>
FPSAN_DEVICE Value<v4f_native, S, Cv>
amdgcn_mfma_scale_f32_16x16x128_f8f6f4(Value<AFrag, S, Cv> a, Value<BFrag, S, Cv> b,
                                       Value<v4f_native, S, Cv> c, int scale_a, int scale_b) {
  if constexpr (S == Semantics::Native) {
    (void)ABID; // scaled MFMA has no ABID operand; kept only for API symmetry.
    auto d = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        __builtin_bit_cast(v8i32_native, a.to_float()),
        __builtin_bit_cast(v8i32_native, b.to_float()), c.to_float(), CBSZ, BLGP, ScaleAOp, scale_a,
        ScaleBOp, scale_b);
    return Value<v4f_native, S, Cv>(d);
  } else {
    static_assert(detail::scaled_fmt_ok<AFrag>(CBSZ) && detail::scaled_fmt_ok<BFrag>(BLGP),
                  "FPSan scaled MFMA: operand fragment element type must match the "
                  "format immediate (E4M3<->0, E5M2<->1); for fp6/bf6/fp4 (2-4) use "
                  "amdgcn_mfma_scale_f32_16x16x128_f8f6f4_sub.");
    return detail::mfma_scale_software_16x16x128(a, b, c, scale_a, scale_b, ScaleAOp, ScaleBOp);
  }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) ||                                                            \
    __has_builtin(__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4)
template <int CBSZ = 0, int ABID = 0, int BLGP = 0, int ScaleAOp = 0, int ScaleBOp = 0,
          Semantics S = Semantics::Native, Conversions Cv = Conversions::Explicit,
          class AFrag = v32e4m3_native, class BFrag = v32e4m3_native>
FPSAN_DEVICE Value<v16f_native, S, Cv>
amdgcn_mfma_scale_f32_32x32x64_f8f6f4(Value<AFrag, S, Cv> a, Value<BFrag, S, Cv> b,
                                      Value<v16f_native, S, Cv> c, int scale_a, int scale_b) {
  if constexpr (S == Semantics::Native) {
    (void)ABID; // scaled MFMA has no ABID operand; kept only for API symmetry.
    auto d = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        __builtin_bit_cast(v8i32_native, a.to_float()),
        __builtin_bit_cast(v8i32_native, b.to_float()), c.to_float(), CBSZ, BLGP, ScaleAOp, scale_a,
        ScaleBOp, scale_b);
    return Value<v16f_native, S, Cv>(d);
  } else {
    static_assert(detail::scaled_fmt_ok<AFrag>(CBSZ) && detail::scaled_fmt_ok<BFrag>(BLGP),
                  "FPSan scaled MFMA: operand fragment element type must match the "
                  "format immediate (E4M3<->0, E5M2<->1); for fp6/bf6/fp4 (2-4) use "
                  "amdgcn_mfma_scale_f32_32x32x64_f8f6f4_sub.");
    return detail::mfma_scale_software_32x32x64(a, b, c, scale_a, scale_b, ScaleAOp, ScaleBOp);
  }
}
#endif

// ---- Sub-byte (fp6 / bf6 / fp4) scaled f8f6f4 MFMA wrappers ----------------
// Companion to the fp8/bf8 wrappers above for the 6-bit (E2M3 fp6 / E3M2 bf6)
// and 4-bit (E2M1 fp4) operand formats. Because sub-byte data is never a Value
// scalar element type, the per-lane operands are passed as the raw packed
// v8i32 register (exactly what the builtin consumes): in Native mode its bits
// are the hardware codes; in FPSan mode they are the per-slot payloads packed
// at the silicon-verified bit positions (Width-bit field s at bits Width*s..).
//
// A and B may have ANY sub-byte widths (fp6/bf6/fp4 in any combination,
// including DIFFERENT widths e.g. fp6 x fp4): all sub-byte formats share one
// physical k ordering, so the hardware pairs equal field indices to the same k
// (silicon-verified full-block-random for every sub x sub combo at both
// shapes). MIXING an
// 8-bit (fp8/bf8) operand WITH a sub-byte one is the one case this wrapper does
// not handle (different physical k orderings); use the *_f8f6f4_mixed wrappers
// below for that.
//
// Layout + scale are the silicon-verified model: the 16x16x128 / 32x32x64 fp8
// lane maps with each byte slot replaced by a Width-bit field, and the same
// per-K-block E8M0 scale (scale_block_factor) as the fp8 path.
#if !defined(__HIP_DEVICE_COMPILE__) ||                                                            \
    __has_builtin(__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4)
template <int CBSZ, int BLGP, int ScaleAOp = 0, int ScaleBOp = 0, Semantics S = Semantics::Native,
          Conversions Cv = Conversions::Explicit>
FPSAN_DEVICE Value<v4f_native, S, Cv>
amdgcn_mfma_scale_f32_16x16x128_f8f6f4_sub(v8i32_native a, v8i32_native b,
                                           Value<v4f_native, S, Cv> c, int scale_a, int scale_b) {
  static_assert(CBSZ >= 2 && BLGP >= 2,
                "the _sub wrapper is for fp6/bf6/fp4 (CBSZ/BLGP 2-4); use the "
                "Value<v32_fragment> wrapper for fp8/bf8 (0,1), or the "
                "*_f8f6f4_mixed wrapper to mix an 8-bit and a sub-byte operand.");
  if constexpr (S == Semantics::Native) {
    auto d = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c.to_float(), CBSZ, BLGP,
                                                              ScaleAOp, scale_a, ScaleBOp, scale_b);
    return Value<v4f_native, S, Cv>(d);
  } else {
    int aw[8], bw[8];
    for (int w = 0; w < 8; ++w) {
      aw[w] = a[w];
      bw[w] = b[w];
    }
    return detail::mfma_scale_sub_16x16x128<detail::f8f6f4_width(CBSZ), detail::f8f6f4_width(BLGP),
                                            S, Cv>(aw, bw, c, scale_a, scale_b, ScaleAOp, ScaleBOp);
  }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) ||                                                            \
    __has_builtin(__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4)
template <int CBSZ, int BLGP, int ScaleAOp = 0, int ScaleBOp = 0, Semantics S = Semantics::Native,
          Conversions Cv = Conversions::Explicit>
FPSAN_DEVICE Value<v16f_native, S, Cv>
amdgcn_mfma_scale_f32_32x32x64_f8f6f4_sub(v8i32_native a, v8i32_native b,
                                          Value<v16f_native, S, Cv> c, int scale_a, int scale_b) {
  static_assert(CBSZ >= 2 && BLGP >= 2,
                "the _sub wrapper is for fp6/bf6/fp4 (CBSZ/BLGP 2-4); use the "
                "Value<v32_fragment> wrapper for fp8/bf8 (0,1), or the "
                "*_f8f6f4_mixed wrapper to mix an 8-bit and a sub-byte operand.");
  if constexpr (S == Semantics::Native) {
    auto d = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c.to_float(), CBSZ, BLGP,
                                                             ScaleAOp, scale_a, ScaleBOp, scale_b);
    return Value<v16f_native, S, Cv>(d);
  } else {
    int aw[8], bw[8];
    for (int w = 0; w < 8; ++w) {
      aw[w] = a[w];
      bw[w] = b[w];
    }
    return detail::mfma_scale_sub_32x32x64<detail::f8f6f4_width(CBSZ), detail::f8f6f4_width(BLGP),
                                           S, Cv>(aw, bw, c, scale_a, scale_b, ScaleAOp, ScaleBOp);
  }
}
#endif

// ---- Mixed-WIDTH (8-bit x sub-byte) scaled f8f6f4 MFMA wrappers ------------
// For pairing an fp8/bf8 operand with an fp6/bf6/fp4 operand. The 8-bit operand
// is a Value<v32_fragment> (proper fp8 payload casting); the sub-byte operand
// is the raw packed v8i32. `_mixed_a8`: A is 8-bit, B is sub-byte; `_mixed_b8`:
// A is sub-byte, B is 8-bit. The two formats use different physical k orderings
// on hardware -- the sub operand follows the silicon-RE'd mix model that pairs
// it with fp8 (detail::mfma_scale_mixed_*; proven full-block-random on
// MI350). Per-K-block E8M0 scale as elsewhere.
#if !defined(__HIP_DEVICE_COMPILE__) ||                                                            \
    __has_builtin(__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4)
template <int CBSZ, int BLGP, int ScaleAOp = 0, int ScaleBOp = 0, Semantics S = Semantics::Native,
          Conversions Cv = Conversions::Explicit, class AFrag = v32e4m3_native>
FPSAN_DEVICE Value<v4f_native, S, Cv> amdgcn_mfma_scale_f32_16x16x128_f8f6f4_mixed_a8(
    Value<AFrag, S, Cv> a, v8i32_native b, Value<v4f_native, S, Cv> c, int scale_a, int scale_b) {
  static_assert(CBSZ <= 1, "mixed_a8: A must be 8-bit fp8/bf8 (CBSZ 0 or 1).");
  static_assert(BLGP >= 2, "mixed_a8: B must be sub-byte fp6/bf6/fp4 (BLGP 2-4); for 8-bit "
                           "x 8-bit use the plain f8f6f4 wrapper.");
  if constexpr (S == Semantics::Native) {
    auto d = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        __builtin_bit_cast(v8i32_native, a.to_float()), b, c.to_float(), CBSZ, BLGP, ScaleAOp,
        scale_a, ScaleBOp, scale_b);
    return Value<v4f_native, S, Cv>(d);
  } else {
    static_assert(detail::scaled_fmt_ok<AFrag>(CBSZ),
                  "mixed_a8: A fragment element type must match CBSZ (E4M3<->0, "
                  "E5M2<->1).");
    int bw[8];
    for (int w = 0; w < 8; ++w)
      bw[w] = b[w];
    return detail::mfma_scale_mixed_16x16x128<false, detail::f8f6f4_width(BLGP), AFrag, S, Cv>(
        a, bw, c, scale_a, scale_b, ScaleAOp, ScaleBOp);
  }
}

template <int CBSZ, int BLGP, int ScaleAOp = 0, int ScaleBOp = 0, Semantics S = Semantics::Native,
          Conversions Cv = Conversions::Explicit, class BFrag = v32e4m3_native>
FPSAN_DEVICE Value<v4f_native, S, Cv> amdgcn_mfma_scale_f32_16x16x128_f8f6f4_mixed_b8(
    v8i32_native a, Value<BFrag, S, Cv> b, Value<v4f_native, S, Cv> c, int scale_a, int scale_b) {
  static_assert(BLGP <= 1, "mixed_b8: B must be 8-bit fp8/bf8 (BLGP 0 or 1).");
  static_assert(CBSZ >= 2, "mixed_b8: A must be sub-byte fp6/bf6/fp4 (CBSZ 2-4); for 8-bit "
                           "x 8-bit use the plain f8f6f4 wrapper.");
  if constexpr (S == Semantics::Native) {
    auto d = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, __builtin_bit_cast(v8i32_native, b.to_float()), c.to_float(), CBSZ, BLGP, ScaleAOp,
        scale_a, ScaleBOp, scale_b);
    return Value<v4f_native, S, Cv>(d);
  } else {
    static_assert(detail::scaled_fmt_ok<BFrag>(BLGP),
                  "mixed_b8: B fragment element type must match BLGP (E4M3<->0, "
                  "E5M2<->1).");
    int aw[8];
    for (int w = 0; w < 8; ++w)
      aw[w] = a[w];
    return detail::mfma_scale_mixed_16x16x128<true, detail::f8f6f4_width(CBSZ), BFrag, S, Cv>(
        b, aw, c, scale_a, scale_b, ScaleAOp, ScaleBOp);
  }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) ||                                                            \
    __has_builtin(__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4)
template <int CBSZ, int BLGP, int ScaleAOp = 0, int ScaleBOp = 0, Semantics S = Semantics::Native,
          Conversions Cv = Conversions::Explicit, class AFrag = v32e4m3_native>
FPSAN_DEVICE Value<v16f_native, S, Cv> amdgcn_mfma_scale_f32_32x32x64_f8f6f4_mixed_a8(
    Value<AFrag, S, Cv> a, v8i32_native b, Value<v16f_native, S, Cv> c, int scale_a, int scale_b) {
  static_assert(CBSZ <= 1, "mixed_a8: A must be 8-bit fp8/bf8 (CBSZ 0 or 1).");
  static_assert(BLGP >= 2, "mixed_a8: B must be sub-byte fp6/bf6/fp4 (BLGP 2-4); for 8-bit "
                           "x 8-bit use the plain f8f6f4 wrapper.");
  if constexpr (S == Semantics::Native) {
    auto d = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        __builtin_bit_cast(v8i32_native, a.to_float()), b, c.to_float(), CBSZ, BLGP, ScaleAOp,
        scale_a, ScaleBOp, scale_b);
    return Value<v16f_native, S, Cv>(d);
  } else {
    static_assert(detail::scaled_fmt_ok<AFrag>(CBSZ),
                  "mixed_a8: A fragment element type must match CBSZ (E4M3<->0, "
                  "E5M2<->1).");
    int bw[8];
    for (int w = 0; w < 8; ++w)
      bw[w] = b[w];
    return detail::mfma_scale_mixed_32x32x64<false, detail::f8f6f4_width(BLGP), AFrag, S, Cv>(
        a, bw, c, scale_a, scale_b, ScaleAOp, ScaleBOp);
  }
}

template <int CBSZ, int BLGP, int ScaleAOp = 0, int ScaleBOp = 0, Semantics S = Semantics::Native,
          Conversions Cv = Conversions::Explicit, class BFrag = v32e4m3_native>
FPSAN_DEVICE Value<v16f_native, S, Cv> amdgcn_mfma_scale_f32_32x32x64_f8f6f4_mixed_b8(
    v8i32_native a, Value<BFrag, S, Cv> b, Value<v16f_native, S, Cv> c, int scale_a, int scale_b) {
  static_assert(BLGP <= 1, "mixed_b8: B must be 8-bit fp8/bf8 (BLGP 0 or 1).");
  static_assert(CBSZ >= 2, "mixed_b8: A must be sub-byte fp6/bf6/fp4 (CBSZ 2-4); for 8-bit "
                           "x 8-bit use the plain f8f6f4 wrapper.");
  if constexpr (S == Semantics::Native) {
    auto d = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a, __builtin_bit_cast(v8i32_native, b.to_float()), c.to_float(), CBSZ, BLGP, ScaleAOp,
        scale_a, ScaleBOp, scale_b);
    return Value<v16f_native, S, Cv>(d);
  } else {
    static_assert(detail::scaled_fmt_ok<BFrag>(BLGP),
                  "mixed_b8: B fragment element type must match BLGP (E4M3<->0, "
                  "E5M2<->1).");
    int aw[8];
    for (int w = 0; w < 8; ++w)
      aw[w] = a[w];
    return detail::mfma_scale_mixed_32x32x64<true, detail::f8f6f4_width(CBSZ), BFrag, S, Cv>(
        b, aw, c, scale_a, scale_b, ScaleAOp, ScaleBOp);
  }
}
#endif

#undef FPSAN_MFMA_EXTRACT_VEC8
#undef FPSAN_MFMA_EXTRACT_VEC16
#undef FPSAN_MFMA_EXTRACT_FP8

} // namespace fpsan

#endif // FPSAN_AMDGCN_MFMA_HPP
