// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/amdgcn_swmmac_gfx1250.hpp
// ----------------------------------------------------------------------------
// FPSan wrappers for the gfx1250 sparse WMMA (SWMMAC) family on wave32:
//
//   swmmac_f32_16x16x128_fp8_fp8 / fp8_bf8 / bf8_fp8 / bf8_bf8
//   swmmac_f16_16x16x128_fp8_fp8 / fp8_bf8 / bf8_fp8 / bf8_bf8
//
// 2:4 structured sparsity: A is half-density (2 of every 4 K positions live)
// compressed into the dense K=64 fp8 fragment (v32 bytes / lane = v8i32 ABI);
// B holds full density at K=128 (v64 bytes / lane = v16i32 ABI); a per-lane
// 64-bit index (v2i32) names, per compressed slot, which of the 4 K positions
// in its group is live (2 bits each).
//
// Native mode: bit-exact pass-through to __builtin_amdgcn_swmmac_*.
//
// FPSan mode: wave-cooperative software dataflow in the payload ring.
//   * A_comp[i][c] (c = compressed K 0..63) uses Swmmac16x16x128CompressedALayout
//     (the SWMMAC compressed-A layout; NOT the dense 16x16x64 A layout).
//   * B[k][j]      (k = dense K 0..127)     uses Wmma16x16x128Layout (the dense
//     16x16x128 fp8 B layout, validated by the dense WMMA suite).
// All of the SWMMAC-specific mappings (compressed-A fragment layout, sparse-index
// selector location, and the base = 4*(c/2) dense grouping) were pinned EMPIRICALLY
// against the hardware builtin (one-hot A + delta/binary-encoded B readback). For
// compressed slot c of output row i: the 2-bit selector lives on
//   lane = i + 16*(c/32), dword = (c/2)&1, bit = 4*((c/4)%8) + 2*(c&1),
// and the live dense K is k = 4*(c/2) + selector. The self-checking layout test
// (real builtin vs this dataflow vs an independent host sparse matmul over
// identically staged fragments) validates the whole mapping end-to-end.
//
// HIP/device-only. Opt-in (not pulled by <fpsan/fpsan.hpp>).
// ----------------------------------------------------------------------------
#ifndef FPSAN_AMDGCN_SWMMAC_GFX1250_HPP
#define FPSAN_AMDGCN_SWMMAC_GFX1250_HPP

#include "fpsan/amdgcn_matrix.hpp"
#include "fpsan/cast.hpp"
#include "fpsan/value.hpp"

#include <cstdint>

#if !defined(__HIP__) && !defined(__CUDACC__)
#error "fpsan/amdgcn_swmmac_gfx1250.hpp is GPU-only; compile as HIP (or CUDA)."
#endif

namespace fpsan {
namespace detail {
// ---- Compressed-A fragment layout for gfx1250 SWMMAC (8-bit, K=128) -------
// The compressed A operand carries 64 K-elements per row in a v32 fp8/bf8
// fragment (8 dwords / lane). It is NOT the dense 16x16x64 A layout. The
// mapping below was determined empirically against the hardware builtin
// using identity selectors + delta-B readback and is pinned by the
// self-validating layout test. For compressed slot cc in 0..63 of output
// row `row`:
//   lane = row + 16*((cc>>1)&1)     // bit1 of cc -> lane half
//   idx  = 2*(cc>>2) + (cc&1)       // (0..31) fragment slot
// i.e. each consecutive pair of cc shares a fragment idx pair, and bit1 of cc
// toggles the lane half: cc 0,1 -> half0 idx0,1 ; cc 2,3 -> half1 idx0,1 ;
// cc 4,5 -> half0 idx2,3 ; cc 6,7 -> half1 idx2,3 ; ...
struct Swmmac16x16x128CompressedALayout {
  FPSAN_DEVICE static int ab_lane(int row, int cc) { return row + 16 * ((cc >> 1) & 1); }
  FPSAN_DEVICE static int ab_index(int cc) { return 2 * (cc >> 2) + (cc & 1); }
};

// ---- fp8 / bf8 SWMMAC 16x16x128 dataflow (K=128 dense, K=64 compressed) ---
// a = compressed A (v32 fp8 / lane), b = dense B (v64 fp8 / lane), c = v8
// accumulator, index = per-lane v2i32 sparse selectors. Generic over
// Semantics and the C/D element type (f32 or f16 accumulate-in-type).
template <class AVec, class BVec, class CVec, Semantics S, Conversions C>
FPSAN_DEVICE Value<CVec, S, C>
swmmac_software_16x16x128_fp8(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c,
                              v2i32_native index) {
  using DFrag = Value<CVec, S, C>;
  using AccScalar = typename DFrag::element_type;
  using Acc = Value<AccScalar, S, C>;
  const int lane = wave_lane();
  const int j = lane & 15;

  // Prefetch this lane's dense B column j (k = 0..127) using the validated
  // dense K=128 B layout. The slot index depends only on k (uniform across
  // the wave), so each wave_shfl is well formed.
  Acc Bcol[128];
  for (int k = 0; k < 128; ++k) {
    const int bslot = Wmma16x16x128Layout::ab_index(k);
    const int blane = Wmma16x16x128Layout::ab_lane(j, k);
    Bcol[k] = cast<AccScalar>(wave_shfl(b.get(bslot), blane));
  }

  DFrag d{};
  for (int e = 0; e < 8; ++e) {
    const int i = e + 8 * (lane >> 4);
    Acc acc = c.get(e);
    for (int cc = 0; cc < 64; ++cc) {
      // Sparse selector for compressed slot cc of output row i. The
      // 2-bit selector location was pinned empirically against the
      // hardware builtin:
      //   lane = i + 16*(cc/32) ; dword = (cc/2)&1 ;
      //   bit  = 4*((cc/4)%8) + 2*(cc&1)
      // and the live dense K is base + selector with base = 4*(cc/2).
      const int idx_lane = i + 16 * (cc / 32);
      const int idx_dword = (cc / 2) & 1;
      const int bit_off = 4 * ((cc / 4) % 8) + 2 * (cc & 1);
      const int idxword = __builtin_amdgcn_ds_bpermute(idx_lane * 4, index[idx_dword]);
      const int sel = (idxword >> bit_off) & 3;
      const int k_dense = 4 * (cc / 2) + sel;
      // Compressed A slot cc uses the SWMMAC compressed-A layout
      // (empirically pinned; distinct from the dense K=64 A layout).
      const int a_slot = Swmmac16x16x128CompressedALayout::ab_index(cc);
      const int a_lane = Swmmac16x16x128CompressedALayout::ab_lane(i, cc);
      auto av = wave_shfl(a.get(a_slot), a_lane);
      acc = acc + cast<AccScalar>(av) * Bcol[k_dense];
    }
    d.set(e, acc);
  }
  return d;
}

// ---- Compressed-A fragment layout for gfx1250 SWMMAC (16-bit, K=64) -------
// The compressed A operand carries 32 K-elements per row in a v16 f16/bf16
// fragment (16 dwords... no: 16 elems / lane). Hypothesis scaled from the
// empirically-pinned 8-bit K=128 compressed-A layout (halved K): for
// compressed slot cc in 0..31 of output row `row`:
//   lane = row + 16*((cc>>1)&1)   // bit1 of cc -> lane half
//   idx  = 2*(cc>>2) + (cc&1)     // (0..15) fragment slot
// Pinned/validated by the self-checking layout test.
struct Swmmac16x16x64CompressedALayout {
  FPSAN_DEVICE static int ab_lane(int row, int cc) { return row + 16 * ((cc >> 1) & 1); }
  FPSAN_DEVICE static int ab_index(int cc) { return 2 * (cc >> 2) + (cc & 1); }
};

// ---- 16-bit (f16/bf16) SWMMAC 16x16x64 dataflow --------------------------
// a = compressed A (v16 / lane, K=32 compressed), b = dense B (v32 / lane,
// K=64 dense), c = v8 accumulator, index = single i32 of sparse selectors.
// Dense B reuses the K=64 element-index layout (Wmma16x16x64Layout); the
// sparse selector for slot cc lives at lane = i + 16*(cc>>4) of the per-lane
// i32, bit = 16*((cc>>1)&1) + 4*((cc>>2)&3) + 2*(cc&1), with live dense
// k = 4*(cc/2) + selector. The compressed-A layout, B-layout reuse, and this
// selector location were all pinned empirically against the hardware builtin
// and are arbitrated by the self-validating test.
template <class AVec, class BVec, class CVec, Semantics S, Conversions C>
FPSAN_DEVICE Value<CVec, S, C> swmmac_software_16x16x64_16(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                                           Value<CVec, S, C> c, int index) {
  using DFrag = Value<CVec, S, C>;
  using AccScalar = typename DFrag::element_type;
  using Acc = Value<AccScalar, S, C>;
  const int lane = wave_lane();
  const int j = lane & 15;

  Acc Bcol[64];
  for (int k = 0; k < 64; ++k) {
    const int bslot = Wmma16x16x64Layout::ab_index(k);
    const int blane = Wmma16x16x64Layout::ab_lane(j, k);
    Bcol[k] = cast<AccScalar>(wave_shfl(b.get(bslot), blane));
  }

  DFrag d{};
  for (int e = 0; e < 8; ++e) {
    const int i = e + 8 * (lane >> 4);
    Acc acc = c.get(e);
    for (int cc = 0; cc < 32; ++cc) {
      const int idx_lane = i + 16 * (cc >> 4);
      const int bit_off = 16 * ((cc >> 1) & 1) + 4 * ((cc >> 2) & 3) + 2 * (cc & 1);
      const int idxword = __builtin_amdgcn_ds_bpermute(idx_lane * 4, index);
      const int sel = (idxword >> bit_off) & 3;
      const int k_dense = 4 * (cc / 2) + sel;
      const int a_slot = Swmmac16x16x64CompressedALayout::ab_index(cc);
      const int a_lane = Swmmac16x16x64CompressedALayout::ab_lane(i, cc);
      auto av = wave_shfl(a.get(a_slot), a_lane);
      acc = acc + cast<AccScalar>(av) * Bcol[k_dense];
    }
    d.set(e, acc);
  }
  return d;
}
} // namespace detail

// ---- FP8 / BF8 SWMMAC wrappers --------------------------------------------
// Builtin signature: (v8i32 A, v16i32 B, v8{f,h} C, v2i32 index, bool, bool)
//   -> v8{f,h}. A = 32 fp8 bytes/lane (compressed K=64), B = 64 fp8 bytes/lane
//   (dense K=128), C/D = v8 accumulator, index = v2i32 sparse selectors.
#define FPSAN_DEFINE_SWMMAC_GFX1250_FP8(NAME, AVec_, BVec_, CVec_, BUILTIN)                        \
  template <Semantics S, Conversions C>                                                            \
  FPSAN_DEVICE Value<CVec_, S, C> NAME(Value<AVec_, S, C> a, Value<BVec_, S, C> b,                 \
                                       Value<CVec_, S, C> c, v2i32_native index) {                 \
    if constexpr (S == Semantics::Native) {                                                        \
      CVec_ d = BUILTIN(__builtin_bit_cast(v8i32_native, a.to_float()),                            \
                        __builtin_bit_cast(v16i32_native, b.to_float()), c.to_float(), index,      \
                        false, false);                                                             \
      return Value<CVec_, S, C>(d);                                                                \
    } else {                                                                                       \
      return detail::swmmac_software_16x16x128_fp8<AVec_, BVec_, CVec_, S, C>(a, b, c, index);     \
    }                                                                                              \
  }

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_swmmac_f32_16x16x128_fp8_fp8)
FPSAN_DEFINE_SWMMAC_GFX1250_FP8(amdgcn_swmmac_f32_16x16x128_fp8_fp8, v32e4m3_native, v64e4m3_native,
                                v8f_native, __builtin_amdgcn_swmmac_f32_16x16x128_fp8_fp8)
FPSAN_DEFINE_SWMMAC_GFX1250_FP8(amdgcn_swmmac_f32_16x16x128_fp8_bf8, v32e4m3_native, v64e5m2_native,
                                v8f_native, __builtin_amdgcn_swmmac_f32_16x16x128_fp8_bf8)
FPSAN_DEFINE_SWMMAC_GFX1250_FP8(amdgcn_swmmac_f32_16x16x128_bf8_fp8, v32e5m2_native, v64e4m3_native,
                                v8f_native, __builtin_amdgcn_swmmac_f32_16x16x128_bf8_fp8)
FPSAN_DEFINE_SWMMAC_GFX1250_FP8(amdgcn_swmmac_f32_16x16x128_bf8_bf8, v32e5m2_native, v64e5m2_native,
                                v8f_native, __builtin_amdgcn_swmmac_f32_16x16x128_bf8_bf8)
FPSAN_DEFINE_SWMMAC_GFX1250_FP8(amdgcn_swmmac_f16_16x16x128_fp8_fp8, v32e4m3_native, v64e4m3_native,
                                v8h_native, __builtin_amdgcn_swmmac_f16_16x16x128_fp8_fp8)
FPSAN_DEFINE_SWMMAC_GFX1250_FP8(amdgcn_swmmac_f16_16x16x128_fp8_bf8, v32e4m3_native, v64e5m2_native,
                                v8h_native, __builtin_amdgcn_swmmac_f16_16x16x128_fp8_bf8)
FPSAN_DEFINE_SWMMAC_GFX1250_FP8(amdgcn_swmmac_f16_16x16x128_bf8_fp8, v32e5m2_native, v64e4m3_native,
                                v8h_native, __builtin_amdgcn_swmmac_f16_16x16x128_bf8_fp8)
FPSAN_DEFINE_SWMMAC_GFX1250_FP8(amdgcn_swmmac_f16_16x16x128_bf8_bf8, v32e5m2_native, v64e5m2_native,
                                v8h_native, __builtin_amdgcn_swmmac_f16_16x16x128_bf8_bf8)
#endif

#undef FPSAN_DEFINE_SWMMAC_GFX1250_FP8

// ---- 16-bit (f16 / bf16) SWMMAC 16x16x64 wrappers -------------------------
// Builtin signature: (bool negA, v16{h,bf} A, bool negB, v32{h,bf} B,
//   v8{f,h,bf} C, int index, bool, bool) -> v8{f,h,bf}. A = 32 elems/lane
//   (compressed K=32), B = 32 elems/lane (dense K=64), C/D = v8 accumulator,
//   index = single i32 of sparse selectors.
#define FPSAN_DEFINE_SWMMAC_GFX1250_16(NAME, AVec_, BVec_, CVec_, BUILTIN)                         \
  template <Semantics S, Conversions C>                                                            \
  FPSAN_DEVICE Value<CVec_, S, C> NAME(Value<AVec_, S, C> a, Value<BVec_, S, C> b,                 \
                                       Value<CVec_, S, C> c, int index) {                          \
    if constexpr (S == Semantics::Native) {                                                        \
      CVec_ d =                                                                                    \
          BUILTIN(false, a.to_float(), false, b.to_float(), c.to_float(), index, false, false);    \
      return Value<CVec_, S, C>(d);                                                                \
    } else {                                                                                       \
      return detail::swmmac_software_16x16x64_16<AVec_, BVec_, CVec_, S, C>(a, b, c, index);       \
    }                                                                                              \
  }

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_swmmac_f32_16x16x64_f16)
FPSAN_DEFINE_SWMMAC_GFX1250_16(amdgcn_swmmac_f32_16x16x64_f16, v16h_native, v32h_native, v8f_native,
                               __builtin_amdgcn_swmmac_f32_16x16x64_f16)
FPSAN_DEFINE_SWMMAC_GFX1250_16(amdgcn_swmmac_f16_16x16x64_f16, v16h_native, v32h_native, v8h_native,
                               __builtin_amdgcn_swmmac_f16_16x16x64_f16)
FPSAN_DEFINE_SWMMAC_GFX1250_16(amdgcn_swmmac_f32_16x16x64_bf16, v16bf_native, v32bf_native,
                               v8f_native, __builtin_amdgcn_swmmac_f32_16x16x64_bf16)
FPSAN_DEFINE_SWMMAC_GFX1250_16(amdgcn_swmmac_bf16_16x16x64_bf16, v16bf_native, v32bf_native,
                               v8bf_native, __builtin_amdgcn_swmmac_bf16_16x16x64_bf16)
FPSAN_DEFINE_SWMMAC_GFX1250_16(amdgcn_swmmac_bf16f32_16x16x64_bf16, v16bf_native, v32bf_native,
                               v8f_native, __builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16)
#endif

#undef FPSAN_DEFINE_SWMMAC_GFX1250_16

} // namespace fpsan

#endif // FPSAN_AMDGCN_SWMMAC_GFX1250_HPP
