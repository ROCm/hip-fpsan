// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/amdgcn_smfmac.hpp
// ----------------------------------------------------------------------------
// FPSan wrappers for AMDGPU SMFMAC (Sparse Matrix Fused-Multiply-ACcumulate)
// intrinsics on gfx950 (CDNA4) and earlier CDNA archs.
//
// Structured 2:4 sparsity: A is half-density -- of every 4 contiguous K
// positions, only 2 are non-zero -- so the same MFMA shape doubles K for
// the same A-fragment register cost.  The dense B is unchanged but holds
// 2x the K elements per lane.  A 16-bit per-lane `index` argument selects
// which 2-of-4 K positions are non-zero in A.
//
// Wrappers below cover the gfx950 SMFMAC family:
//   smfmac_f32_16x16x64_{f16,bf16}              (gfx950-insts)
//   smfmac_f32_32x32x32_{f16,bf16}              (gfx950-insts)
//   smfmac_f32_16x16x128_{fp8,bf8}_{fp8,bf8}    (gfx950-insts, 4 combos)
//   smfmac_f32_32x32x64_{fp8,bf8}_{fp8,bf8}     (gfx950-insts, 4 combos)
//
// Plus the older CDNA shapes still present on gfx950 hardware:
//   smfmac_f32_16x16x32_{f16,bf16}              (mai-insts)
//   smfmac_f32_32x32x16_{f16,bf16}              (mai-insts)
//   smfmac_f32_16x16x64_{fp8,bf8}_{fp8,bf8}     (fp8-insts)
//   smfmac_f32_32x32x32_{fp8,bf8}_{fp8,bf8}     (fp8-insts)
//
// Native mode forwards to the matching __builtin_amdgcn_smfmac_*.  FPSan mode
// runs the sparse software dataflow: it decodes the per-lane `index` operand
// (2-bit selectors naming the 2-of-4 live K positions) and accumulates over the
// live positions in the payload ring. Implemented + silicon-verified for EVERY
// SMFMAC shape in the family: all four f16/bf16 shapes (gfx950-inst 16x16x64 /
// 32x32x32 and CDNA3 mai-inst 16x16x32 / 32x32x16) AND all fp8/bf8 shapes
// (CDNA3 fp8-inst 16x16x64 / 32x32x32 and gfx950-inst 16x16x128 / 32x32x64),
// see smfmac_software_*.  We deliberately never silently return the
// accumulator; any not-yet-proven shape would be a HARD COMPILE ERROR.
//
// Wave64.  HIP/device-only.  Opt-in (not pulled by <fpsan/fpsan.hpp>).
// ----------------------------------------------------------------------------
#ifndef FPSAN_AMDGCN_SMFMAC_HPP
#define FPSAN_AMDGCN_SMFMAC_HPP

#include "fpsan/amdgcn_matrix.hpp"
#include "fpsan/amdgcn_mfma.hpp" // v4f_native / v16f_native / v8e4m3_native
#include "fpsan/value.hpp"

#include <cstdint>
#include <type_traits>

#if !defined(__HIP__) && !defined(__CUDACC__)
#error "fpsan/amdgcn_smfmac.hpp is GPU-only; compile as HIP (or CUDA)."
#endif

namespace fpsan {

// gfx950 SMFMAC native fragment vector aliases: B is 2x wider than the
// dense MFMA's B (because sparsity doubles K), so we need vec16 of the
// element type.
using v16h_native = _Float16 __attribute__((ext_vector_type(16)));
using v16bf_native = __bf16 __attribute__((ext_vector_type(16)));
// CDNA3 SMFMAC (K=32/16) uses smaller B: 4-elem half/bf16 for A side, 8 elem
// for B side.
using v4h_native = _Float16 __attribute__((ext_vector_type(4)));
using v4bf_native = __bf16 __attribute__((ext_vector_type(4)));

// FP8 SMFMAC fragment aliases. CDNA3 fp8-insts: A = 8 fp8 bytes (v8e4m3/e5m2,
// already defined in amdgcn_matrix.hpp), B = 16 fp8 bytes (v16). gfx950-insts:
// A = 16 fp8 bytes (v16), B = 32 fp8 bytes (v32, defined in amdgcn_mfma.hpp).
// (v16e4m3_native / v16e5m2_native come from amdgcn_mfma.hpp.)

namespace detail {

// ---- Sparse (2:4) software dataflow: SMFMAC F32_16x16x64_{f16,bf16} --------
// D = C + A_sparse * B, A is 2:4 structured-sparse (2 of every 4 K nonzero),
// stored 2:1-compressed as A_comp[16][32]. All layouts reverse-engineered +
// ISA-cross-checked on real MI350 (see header note):
//   * A_comp[i][c] @ lane (c/8)*16+i, half c%8  (== the dense 16x16x32 layout)
//   * B[k][j]      @ lane j+16*((k%32)/8), vreg (k%8)/2 + (k>=32?4:0), half k%2
//   * D[i][j]      @ output_loc_32(16,16,...): reg i%4, lane (i/4)*16+j
//   * index: per lane, low 16 bits = 4 nibbles for that lane's 4 groups; group
//     g's nibble gives the two live K offsets idx0=nib&3, idx1=(nib>>2)&3.
//     Lane (q/4)*16+i holds the nibble for row i, group q at nibble q%4.
// ds_bpermute moves each source lane's *own* contributed scalar, so every
// cross-lane gather must use an element index that is uniform across the wave.
// The A gather (element = 2q+s, depends only on the loop) and the index gather
// are uniform and direct. The B element index, however, is 2*vr+h with k=4q+p
// and p coming from each lane's sparse index -- per-lane, NOT uniform. So we
// first prefetch this lane's whole B column B[0..63][j] using uniform-element
// shuffles (the 4 lanes sharing column j collectively hold all 64), then index
// it locally by the per-lane k.
template <class AVec, class BVec, Semantics S, Conversions C>
FPSAN_DEVICE Value<v4f_native, S, C> smfmac_software_16x16x64(Value<AVec, S, C> a,
                                                              Value<BVec, S, C> b,
                                                              Value<v4f_native, S, C> c, int idx) {
  using AccF = Value<float, S, C>;
  const int lane = wave_lane_full();
  const int j = lane % 16;
  // Prefetch B[k][j] for all k. Lane g*16+j holds the 16 elements e=2*vr+h with
  // k = (e<8 ? 8g+e : 32+8g+(e-8)). e is uniform across lanes; the source lane
  // g*16+j is per-lane (column j) -- both legal for ds_bpermute.
  AccF Bcol[64];
  for (int g = 0; g < 4; ++g)
    for (int e = 0; e < 16; ++e) {
      auto bv = wave_shfl(b.get(e), g * 16 + j);
      const int k = (e < 8) ? (8 * g + e) : (32 + 8 * g + (e - 8));
      Bcol[k] = cast<float>(bv);
    }
  Value<v4f_native, S, C> d{};
  for (int reg = 0; reg < 4; ++reg) {
    const int i = 4 * (lane / 16) + reg; // output_loc_32 inverse
    AccF acc = c.get(reg);
    for (int q = 0; q < 16; ++q) {
      const int laneA = (q / 4) * 16 + i;
      // Pull the sparse index for (row i, group q) from the lane that holds it.
      const int idxA = __builtin_amdgcn_ds_bpermute(laneA * 4, idx);
      const int nib = (idxA >> (4 * (q % 4))) & 0xF;
      const int p[2] = {nib & 3, (nib >> 2) & 3};
      const int half[2] = {(2 * q) % 8, (2 * q + 1) % 8};
      for (int s = 0; s < 2; ++s) {
        auto av = wave_shfl(a.get(half[s]), laneA);
        acc = acc + cast<float>(av) * Bcol[4 * q + p[s]];
      }
    }
    d.set(reg, acc);
  }
  return d;
}

// ---- Sparse (2:4) software dataflow: SMFMAC F32_32x32x32_{f16,bf16} --------
// K=32 logical (16 compressed), 8 groups. Layouts verified on MI350:
//   * A_comp[i][c] @ lane (c/8)*32+i, half c%8  (== the dense 32x32x16 layout)
//   * B[k][j] @ lane 16*g + (j%16), g = 2*((k%16)/8) + (j/16); register element
//     2*vreg + (k%2), vreg = (k%8)/2 + 4*(k/16). Column j lives on lanes j and
//     j+32 (16 K-values each).
//   * D[i][j] @ output_loc_32(32,32,...), 16 regs/lane.
//   * index: per lane, low 16 bits = 4 nibbles for the lane's 4 groups; lane
//     (q/4)*32+i carries row i, group q at nibble q%4 (idx0=nib&3,
//     idx1=nib>>2).
// As in the 16x16x64 case, the B element index is per-lane, so prefetch this
// lane's whole B column with uniform-element shuffles, then index locally.
template <class AVec, class BVec, Semantics S, Conversions C>
FPSAN_DEVICE Value<v16f_native, S, C>
smfmac_software_32x32x32(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<v16f_native, S, C> c,
                         int idx) {
  using AccF = Value<float, S, C>;
  const int lane = wave_lane_full();
  const int j = lane % 32;
  AccF Bcol[32];
  for (int sl = 0; sl < 2; ++sl) {
    const int src = j + 32 * sl;
    const int kgrp = (src / 16 - j / 16) / 2; // 0 or 1
    for (int e = 0; e < 16; ++e) {
      const int k = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
      Bcol[k] = cast<float>(wave_shfl(b.get(e), src));
    }
  }
  Value<v16f_native, S, C> d{};
  for (int reg = 0; reg < 16; ++reg) {
    int oi = -1; // row for this output reg; column is j (== lane%32)
    for (int i = 0; i < 32 && oi < 0; ++i)
      for (int jj = 0; jj < 32; ++jj) {
        OutputLoc loc = output_loc_32(32, 32, i, jj, 0);
        if (loc.reg == reg && loc.lane == lane) {
          oi = i;
          break;
        }
      }
    if (oi < 0)
      continue;
    AccF acc = c.get(reg);
    for (int q = 0; q < 8; ++q) {
      const int laneA = (q / 4) * 32 + oi;
      const int idxA = __builtin_amdgcn_ds_bpermute(laneA * 4, idx);
      const int nib = (idxA >> (4 * (q % 4))) & 0xF;
      const int p[2] = {nib & 3, (nib >> 2) & 3};
      const int half[2] = {(2 * q) % 8, (2 * q + 1) % 8};
      for (int s = 0; s < 2; ++s) {
        auto av = wave_shfl(a.get(half[s]), laneA);
        acc = acc + cast<float>(av) * Bcol[4 * q + p[s]];
      }
    }
    d.set(reg, acc);
  }
  return d;
}

// ---- Sparse (2:4) software dataflow: SMFMAC F32_16x16x32_{f16,bf16} --------
// CDNA3 mai-inst shape. K=32 logical (16 compressed), 8 groups. A is v4 half,
// B is v8 half. All layouts reverse-engineered + confirmed on real MI350 by an
// exhaustive single-hot probe + full-random LayoutMatchesHardware test:
//   * A_comp[i][c] @ lane (c/4)*16+i, half c%4   (== the dense 16x16x16 layout)
//   * B[k][j]      @ lane (k/8)*16+j, half k%8
//   * D[i][j]      @ output_loc_32(16,16,...): reg i%4, lane (i/4)*16+j
//   * index: per lane, low 16 bits hold the selectors for 2 groups (nibbles 0,1
//     == a full 4-bit p0|p1 field each). Lane (q/2)*16+i carries (row i, group
//     q) at nibble q%2; p0 = field&3, p1 = (field>>2)&3.
// NB: unlike the gfx950-inst K=64 shape, here the index packs 2 groups/lane
// (not 4) and A's two columns of a group share lane (q/2)*16+i (not
// (q/4)*16+i). As in the K=64 case, the B element index is per-lane, so
// prefetch this lane's whole B column with uniform-element shuffles, then index
// locally.
template <class AVec, class BVec, Semantics S, Conversions C>
FPSAN_DEVICE Value<v4f_native, S, C> smfmac_software_16x16x32(Value<AVec, S, C> a,
                                                              Value<BVec, S, C> b,
                                                              Value<v4f_native, S, C> c, int idx) {
  using AccF = Value<float, S, C>;
  const int lane = wave_lane_full();
  const int j = lane % 16;
  // Prefetch B[k][j] for all 32 k. Column j lives on lanes g*16+j (g=0..3),
  // each holding k = 8g+e (e = uniform element 0..7).
  AccF Bcol[32];
  for (int g = 0; g < 4; ++g)
    for (int e = 0; e < 8; ++e)
      Bcol[8 * g + e] = cast<float>(wave_shfl(b.get(e), g * 16 + j));
  Value<v4f_native, S, C> d{};
  for (int reg = 0; reg < 4; ++reg) {
    const int i = 4 * (lane / 16) + reg; // output_loc_32 inverse
    AccF acc = c.get(reg);
    for (int q = 0; q < 8; ++q) {
      const int laneA = (q / 2) * 16 + i; // also the index lane for this group
      const int idxA = __builtin_amdgcn_ds_bpermute(laneA * 4, idx);
      const int field = (idxA >> (4 * (q % 2))) & 0xF;
      const int p[2] = {field & 3, (field >> 2) & 3};
      for (int s = 0; s < 2; ++s) {
        auto av = wave_shfl(a.get((2 * q + s) % 4), laneA);
        acc = acc + cast<float>(av) * Bcol[4 * q + p[s]];
      }
    }
    d.set(reg, acc);
  }
  return d;
}

// ---- Sparse (2:4) software dataflow: SMFMAC F32_32x32x16_{f16,bf16} --------
// CDNA3 mai-inst shape. K=16 logical (8 compressed), 4 groups. A v4 half, B v8
// half, C/D v16 f32. All layouts reverse-engineered + confirmed on real MI350
// (single-hot probe + multi-seed full-random LayoutMatchesHardware):
//   * A_comp[i][c] @ lane (c/4)*32+i, half c%4
//   * B[k][j] @ lane with jcol=(lane%16)+16*((lane/16)%2), kgrp=(lane/16)/2,
//     k=8*kgrp+e; column j lives on lanes 16*jhi+jlow (kgrp0) and 16*(jhi+2)+
//     jlow (kgrp1), jlow=j%16, jhi=j/16.  (== the K-halved 32x32x32 B layout.)
//   * D[i][j] @ output_loc_32(32,32,...), 16 regs/lane.
//   * index: lane (q/2)*32+i carries (row i, group q) at nibble q%2;
//   p0=field&3,
//     p1=field>>2 (2 groups/lane, like the 16x16x32 shape).
template <class AVec, class BVec, Semantics S, Conversions C>
FPSAN_DEVICE Value<v16f_native, S, C>
smfmac_software_32x32x16(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<v16f_native, S, C> c,
                         int idx) {
  using AccF = Value<float, S, C>;
  const int lane = wave_lane_full();
  const int j = lane % 32;
  const int jlow = j % 16, jhi = j / 16;
  AccF Bcol[16];
  for (int kgrp = 0; kgrp < 2; ++kgrp) {
    const int src = 16 * (jhi + 2 * kgrp) + jlow;
    for (int e = 0; e < 8; ++e)
      Bcol[8 * kgrp + e] = cast<float>(wave_shfl(b.get(e), src));
  }
  Value<v16f_native, S, C> d{};
  for (int reg = 0; reg < 16; ++reg) {
    int oi = -1; // row for this output reg; column is j (== lane%32)
    for (int i = 0; i < 32 && oi < 0; ++i)
      for (int jj = 0; jj < 32; ++jj) {
        OutputLoc loc = output_loc_32(32, 32, i, jj, 0);
        if (loc.reg == reg && loc.lane == lane) {
          oi = i;
          break;
        }
      }
    if (oi < 0)
      continue;
    AccF acc = c.get(reg);
    for (int q = 0; q < 4; ++q) {
      const int laneA = (q / 2) * 32 + oi; // also the index lane for this group
      const int idxA = __builtin_amdgcn_ds_bpermute(laneA * 4, idx);
      const int field = (idxA >> (4 * (q % 2))) & 0xF;
      const int p[2] = {field & 3, (field >> 2) & 3};
      for (int s = 0; s < 2; ++s) {
        auto av = wave_shfl(a.get((2 * q + s) % 4), laneA);
        acc = acc + cast<float>(av) * Bcol[4 * q + p[s]];
      }
    }
    d.set(reg, acc);
  }
  return d;
}

// ---- Sparse (2:4) software dataflow: SMFMAC FP8 16x16x64 (CDNA3 fp8-insts) -
// K=64 logical (32 compressed), 16 groups. A = 8 fp8 bytes/lane (v8), B = 16
// fp8 bytes/lane (v16), C/D = v4 f32. Layouts reverse-engineered + multi-seed
// full-random verified on MI350 (single-hot probe). They differ from the f16
// 16x16x64 shape because fp8 packs 4 bytes/dword:
//   * A_comp[i][c] @ lane g*16+i (g=lane/16), byte b; c = 16*(b/4)+4*g+(b%4).
//     Inverse: g=(c%16)/4, b=4*(c/16)+(c%16)%4.
//   * B[k][j] @ lane g*16+j, byte e; k = 32*(e/8)+8*g+(e%8)  (== the f16
//     16x16x64 B element->k mapping; B is element-index identical).
//   * D[i][j] @ output_loc_32(16,16).
//   * index: lane ((q%8)/2)*16+i carries (row i, group q) at nibble
//     2*(q/8)+(q%2); p0=field&3, p1=(field>>2)&3.
template <class AVec, class BVec, Semantics S, Conversions C>
FPSAN_DEVICE Value<v4f_native, S, C>
smfmac_software_16x16x64_fp8(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<v4f_native, S, C> c,
                             int idx) {
  using AccF = Value<float, S, C>;
  const int lane = wave_lane_full();
  const int j = lane % 16;
  AccF Bcol[64];
  for (int g = 0; g < 4; ++g)
    for (int e = 0; e < 16; ++e) {
      const int k = 32 * (e / 8) + 8 * g + (e % 8);
      Bcol[k] = cast<float>(wave_shfl(b.get(e), g * 16 + j));
    }
  Value<v4f_native, S, C> d{};
  for (int reg = 0; reg < 4; ++reg) {
    const int i = 4 * (lane / 16) + reg;
    AccF acc = c.get(reg);
    for (int q = 0; q < 16; ++q) {
      const int ga = ((2 * q) % 16) / 4;
      const int laneA = ga * 16 + i;
      const int idxlane = ((q % 8) / 2) * 16 + i;
      const int nb = 2 * (q / 8) + (q % 2);
      const int idxA = __builtin_amdgcn_ds_bpermute(idxlane * 4, idx);
      const int field = (idxA >> (4 * nb)) & 0xF;
      const int p[2] = {field & 3, (field >> 2) & 3};
      for (int s = 0; s < 2; ++s) {
        const int cc = 2 * q + s;
        const int byte = 4 * (cc / 16) + (cc % 16) % 4;
        auto av = wave_shfl(a.get(byte), laneA);
        acc = acc + cast<float>(av) * Bcol[4 * q + p[s]];
      }
    }
    d.set(reg, acc);
  }
  return d;
}

// ---- Sparse (2:4) software dataflow: SMFMAC FP8 32x32x32 (CDNA3 fp8-insts) -
// K=32 logical (16 compressed), 8 groups. A = v8 fp8, B = v16 fp8, C/D = v16
// f32. Layouts reverse-engineered + verified on MI350:
//   * A_comp[i][c] @ lane g*32+i (g=lane/32), byte b; c = 8*(b/4)+4*g+(b%4).
//     Inverse: g=(c%8)/4, b=4*(c/8)+(c%8)%4.
//   * B[k][j] @ the f16-32x32x32 B layout (element-index identical): column j
//     on lanes jhi+2*kgrp, byte e with k=16*(e/8)+8*kgrp+2*((e/2)%4)+(e%2).
//   * D[i][j] @ output_loc_32(32,32).
//   * index: lane ((q%4)/2)*32+i carries (row i, group q) at nibble
//     2*(q/4)+(q%2).
template <class AVec, class BVec, Semantics S, Conversions C>
FPSAN_DEVICE Value<v16f_native, S, C>
smfmac_software_32x32x32_fp8(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<v16f_native, S, C> c,
                             int idx) {
  using AccF = Value<float, S, C>;
  const int lane = wave_lane_full();
  const int j = lane % 32;
  AccF Bcol[32];
  for (int sl = 0; sl < 2; ++sl) {
    const int src = j + 32 * sl;
    const int kgrp = (src / 16 - j / 16) / 2; // 0 or 1
    for (int e = 0; e < 16; ++e) {
      const int k = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
      Bcol[k] = cast<float>(wave_shfl(b.get(e), src));
    }
  }
  Value<v16f_native, S, C> d{};
  for (int reg = 0; reg < 16; ++reg) {
    int oi = -1;
    for (int i = 0; i < 32 && oi < 0; ++i)
      for (int jj = 0; jj < 32; ++jj) {
        OutputLoc loc = output_loc_32(32, 32, i, jj, 0);
        if (loc.reg == reg && loc.lane == lane) {
          oi = i;
          break;
        }
      }
    if (oi < 0)
      continue;
    AccF acc = c.get(reg);
    for (int q = 0; q < 8; ++q) {
      const int ga = ((2 * q) % 8) / 4;
      const int laneA = ga * 32 + oi;
      const int idxlane = ((q % 4) / 2) * 32 + oi;
      const int nb = 2 * (q / 4) + (q % 2);
      const int idxA = __builtin_amdgcn_ds_bpermute(idxlane * 4, idx);
      const int field = (idxA >> (4 * nb)) & 0xF;
      const int p[2] = {field & 3, (field >> 2) & 3};
      for (int s = 0; s < 2; ++s) {
        const int cc = 2 * q + s;
        const int byte = 4 * (cc / 8) + (cc % 8) % 4;
        auto av = wave_shfl(a.get(byte), laneA);
        acc = acc + cast<float>(av) * Bcol[4 * q + p[s]];
      }
    }
    d.set(reg, acc);
  }
  return d;
}

// ---- Sparse (2:4) software dataflow: SMFMAC FP8 16x16x128 (gfx950-insts) ----
// K=128 logical (64 compressed), 32 groups. A = 16 fp8 bytes/lane (v16), B = 32
// fp8 bytes/lane (v32), C/D = v4 f32. Layouts reverse-engineered + multi-seed
// full-random verified on MI350 (single-hot probe + LayoutMatchesHardware):
//   * A_comp[i][c] @ lane g*16+i, byte 4*hb+(c&3); where, from c's bits,
//     g  = 2*((c>>5)&1) + ((c>>3)&1)   (lane group, 0..3)
//     hb = 2*((c>>2)&1) + ((c>>4)&1)   (high byte index, 0..3, byte=4*hb+c&3)
//   * B[k][j] @ lane g*16+j, byte e; k = 32*(e/8)+8*g+(e%8)  (== the f16
//     16x16x64 B element->k mapping, doubled to 32 elems / 128 K).
//   * D[i][j] @ output_loc_32(16,16): reg i%4, lane (i/4)*16+j.
//   * index: (row i, group q) controlled by nibble `nb` of lane idxlane, with
//       idxlane = 16*(2*(q/16) + ((q/4)%2)) + i
//       nb      = 2*((q/8)%2) + 4*((q%4)/2) + ((q%4)%2)
//     s=0 reads the low 2 bits (p0=field&3), s=1 the high 2 bits (p1=field>>2).
//     dense K position = 4q + p[s].
template <class AVec, class BVec, Semantics S, Conversions C>
FPSAN_DEVICE Value<v4f_native, S, C>
smfmac_software_16x16x128_fp8(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<v4f_native, S, C> c,
                              int idx) {
  using AccF = Value<float, S, C>;
  const int lane = wave_lane_full();
  const int j = lane % 16;
  AccF Bcol[128];
  for (int g = 0; g < 4; ++g)
    for (int e = 0; e < 32; ++e) {
      const int k = 32 * (e / 8) + 8 * g + (e % 8);
      Bcol[k] = cast<float>(wave_shfl(b.get(e), g * 16 + j));
    }
  Value<v4f_native, S, C> d{};
  for (int reg = 0; reg < 4; ++reg) {
    const int i = 4 * (lane / 16) + reg;
    AccF acc = c.get(reg);
    for (int q = 0; q < 32; ++q) {
      const int idxlane = 16 * (2 * (q / 16) + ((q / 4) % 2)) + i;
      const int nb = 2 * ((q / 8) % 2) + 4 * ((q % 4) / 2) + ((q % 4) % 2);
      const int idxA = __builtin_amdgcn_ds_bpermute(idxlane * 4, idx);
      const int field = (idxA >> (4 * nb)) & 0xF;
      const int p[2] = {field & 3, (field >> 2) & 3};
      for (int s = 0; s < 2; ++s) {
        const int cc = 2 * q + s;
        const int ga = 2 * ((cc >> 5) & 1) + ((cc >> 3) & 1);
        const int hb = 2 * ((cc >> 2) & 1) + ((cc >> 4) & 1);
        const int byte = 4 * hb + (cc & 3);
        auto av = wave_shfl(a.get(byte), ga * 16 + i);
        acc = acc + cast<float>(av) * Bcol[4 * q + p[s]];
      }
    }
    d.set(reg, acc);
  }
  return d;
}

// ---- Sparse (2:4) software dataflow: SMFMAC FP8 32x32x64 (gfx950-insts) -----
// K=64 logical (32 compressed), 16 groups. A = 16 fp8 bytes/lane (v16), B = 32
// fp8 bytes/lane (v32), C/D = v16 f32. Layouts reverse-engineered + multi-seed
// full-random verified on MI350 (single-hot probe + LayoutMatchesHardware):
//   * A_comp[i][c] @ lane g*32+i (g=(c>>4)&1), byte 4*hb+(c&3) with
//     hb = 2*((c>>2)&1) + ((c>>3)&1).
//   * B[k][j] @ lane kgrp*32+j (kgrp=lane/32), byte e; k =
//   16*(e/8)+8*kgrp+(e%8)
//     (== the f16/fp8 32x32x32 B element->k mapping, e extended to 0..31).
//   * D[i][j] @ output_loc_32(32,32): reg 4*(i/8)+(i%4), lane 32*((i/4)%2)+j.
//   * index: (row i, group q) controlled by nibble `nb` of lane idxlane, with
//       idxlane = 32*(q/8) + i
//       nb      = (q%2) + 2*((q/4)%2) + 4*((q/2)%2)
//     s=0 reads p0=field&3, s=1 reads p1=field>>2; dense K = 4q + p[s].
template <class AVec, class BVec, Semantics S, Conversions C>
FPSAN_DEVICE Value<v16f_native, S, C>
smfmac_software_32x32x64_fp8(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<v16f_native, S, C> c,
                             int idx) {
  using AccF = Value<float, S, C>;
  const int lane = wave_lane_full();
  const int j = lane % 32;
  AccF Bcol[64];
  for (int kgrp = 0; kgrp < 2; ++kgrp) {
    const int src = kgrp * 32 + j;
    for (int e = 0; e < 32; ++e) {
      const int k = 16 * (e / 8) + 8 * kgrp + (e % 8);
      Bcol[k] = cast<float>(wave_shfl(b.get(e), src));
    }
  }
  Value<v16f_native, S, C> d{};
  for (int reg = 0; reg < 16; ++reg) {
    const int i = 8 * (reg / 4) + (reg % 4) + 4 * (lane / 32);
    AccF acc = c.get(reg);
    for (int q = 0; q < 16; ++q) {
      const int idxlane = 32 * (q / 8) + i;
      const int nb = (q % 2) + 2 * ((q / 4) % 2) + 4 * ((q / 2) % 2);
      const int idxA = __builtin_amdgcn_ds_bpermute(idxlane * 4, idx);
      const int field = (idxA >> (4 * nb)) & 0xF;
      const int p[2] = {field & 3, (field >> 2) & 3};
      for (int s = 0; s < 2; ++s) {
        const int cc = 2 * q + s;
        const int ga = (cc >> 4) & 1;
        const int hb = 2 * ((cc >> 2) & 1) + ((cc >> 3) & 1);
        const int byte = 4 * hb + (cc & 3);
        auto av = wave_shfl(a.get(byte), ga * 32 + i);
        acc = acc + cast<float>(av) * Bcol[4 * q + p[s]];
      }
    }
    d.set(reg, acc);
  }
  return d;
}

} // namespace detail

// =============================================================================
// SMFMAC wrappers.  Native-mode forwards to the builtin.  FPSan-mode runs the
// sparse software dataflow (all four CDNA f16/bf16 shapes + the CDNA3 fp8
// shapes) or, for
// shapes not yet ported, is a hard compile error in FPSan mode -- see note.
// =============================================================================

// 16x16x64 / 32x32x32 with f16 / bf16, gfx950-insts (A: v8 half-density).
#define FPSAN_DEFINE_SMFMAC_F16_GFX950(NAME, M_, N_, K_, AVec_, BVec_, CFragVec_, BUILTIN)         \
  template <int CBSZ = 0, int ABID = 0, Semantics S = Semantics::Native,                           \
            Conversions Cv = Conversions::Explicit>                                                \
  FPSAN_DEVICE Value<CFragVec_, S, Cv> NAME(Value<AVec_, S, Cv> a, Value<BVec_, S, Cv> b,          \
                                            Value<CFragVec_, S, Cv> c, int idx) {                  \
    if constexpr (S == Semantics::Native) {                                                        \
      auto d = BUILTIN(a.to_float(), b.to_float(), c.to_float(), idx, CBSZ, ABID);                 \
      return Value<CFragVec_, S, Cv>(d);                                                           \
    } else if constexpr (M_ == 16 && N_ == 16 && K_ == 64) {                                       \
      return detail::smfmac_software_16x16x64<AVec_, BVec_, S, Cv>(a, b, c, idx);                  \
    } else if constexpr (M_ == 32 && N_ == 32 && K_ == 32) {                                       \
      return detail::smfmac_software_32x32x32<AVec_, BVec_, S, Cv>(a, b, c, idx);                  \
    } else if constexpr (M_ == 16 && N_ == 16 && K_ == 32) {                                       \
      return detail::smfmac_software_16x16x32<AVec_, BVec_, S, Cv>(a, b, c, idx);                  \
    } else if constexpr (M_ == 32 && N_ == 32 && K_ == 16) {                                       \
      return detail::smfmac_software_32x32x16<AVec_, BVec_, S, Cv>(a, b, c, idx);                  \
    } else {                                                                                       \
      /* No remaining f16/bf16 SMFMAC shape should reach here. Refuse to                           \
       * compile rather than silently return the accumulator unchanged. */                         \
      static_assert(fpsan::detail::always_false<std::integral_constant<Semantics, S>>,             \
                    "fpsan: SMFMAC FPSan dataflow for this shape is not implemented "              \
                    "yet; the wrapper is intentionally a hard error in FPSan mode so "             \
                    "it "                                                                          \
                    "cannot return a silently-wrong result. Use Native mode, or wait "             \
                    "for "                                                                         \
                    "the ported shape.");                                                          \
      return c;                                                                                    \
    }                                                                                              \
  }

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_smfmac_f32_16x16x64_f16)
FPSAN_DEFINE_SMFMAC_F16_GFX950(amdgcn_smfmac_f32_16x16x64_f16, 16, 16, 64, v8h_native, v16h_native,
                               v4f_native, __builtin_amdgcn_smfmac_f32_16x16x64_f16)
FPSAN_DEFINE_SMFMAC_F16_GFX950(amdgcn_smfmac_f32_16x16x64_bf16, 16, 16, 64, v8bf_native,
                               v16bf_native, v4f_native, __builtin_amdgcn_smfmac_f32_16x16x64_bf16)
FPSAN_DEFINE_SMFMAC_F16_GFX950(amdgcn_smfmac_f32_32x32x32_f16, 32, 32, 32, v8h_native, v16h_native,
                               v16f_native, __builtin_amdgcn_smfmac_f32_32x32x32_f16)
FPSAN_DEFINE_SMFMAC_F16_GFX950(amdgcn_smfmac_f32_32x32x32_bf16, 32, 32, 32, v8bf_native,
                               v16bf_native, v16f_native, __builtin_amdgcn_smfmac_f32_32x32x32_bf16)
#endif

// CDNA3 shapes (mai-insts) still present on gfx950: K=32/16 with vec4 A and
// vec8 B (half the K of the gfx950-insts shapes above).
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_smfmac_f32_16x16x32_f16)
FPSAN_DEFINE_SMFMAC_F16_GFX950(amdgcn_smfmac_f32_16x16x32_f16, 16, 16, 32, v4h_native, v8h_native,
                               v4f_native, __builtin_amdgcn_smfmac_f32_16x16x32_f16)
FPSAN_DEFINE_SMFMAC_F16_GFX950(amdgcn_smfmac_f32_16x16x32_bf16, 16, 16, 32, v4bf_native,
                               v8bf_native, v4f_native, __builtin_amdgcn_smfmac_f32_16x16x32_bf16)
FPSAN_DEFINE_SMFMAC_F16_GFX950(amdgcn_smfmac_f32_32x32x16_f16, 32, 32, 16, v4h_native, v8h_native,
                               v16f_native, __builtin_amdgcn_smfmac_f32_32x32x16_f16)
FPSAN_DEFINE_SMFMAC_F16_GFX950(amdgcn_smfmac_f32_32x32x16_bf16, 32, 32, 16, v4bf_native,
                               v8bf_native, v16f_native, __builtin_amdgcn_smfmac_f32_32x32x16_bf16)
#endif

#undef FPSAN_DEFINE_SMFMAC_F16_GFX950

// FP8 SMFMAC.  The builtin ABI passes A/B as packed i32 vectors (the byte
// storage is opaque); the wrapper bit-casts the Value fp8 fragment to that
// ABI.  FPSan mode runs the per-shape software dataflow.  CDNA3/gfx94x uses
// AMD FNUZ FP8/BF8 encodings, while gfx950 uses the OCP encodings.
//
// CDNA3 fp8-insts (K=64 16x16, K=32 32x32): A = v8 fp8 (=v2i), B = v16 fp8
// (=v4i).  Both shapes are silicon-verified (smfmac_software_*_fp8).
using v2i32_smf = int __attribute__((ext_vector_type(2)));
using v4i32_smf = int __attribute__((ext_vector_type(4)));
using v8i32_smf = int __attribute__((ext_vector_type(8)));

#define FPSAN_DEFINE_SMFMAC_FP8(NAME, AVec_, BVec_, CFragVec_, AABI_, BABI_, SOFT_, BUILTIN)       \
  template <int CBSZ = 0, int ABID = 0, Semantics S = Semantics::Native,                           \
            Conversions Cv = Conversions::Explicit>                                                \
  FPSAN_DEVICE Value<CFragVec_, S, Cv> NAME(Value<AVec_, S, Cv> a, Value<BVec_, S, Cv> b,          \
                                            Value<CFragVec_, S, Cv> c, int idx) {                  \
    if constexpr (S == Semantics::Native) {                                                        \
      const AABI_ ai = __builtin_bit_cast(AABI_, a.to_float());                                    \
      const BABI_ bi = __builtin_bit_cast(BABI_, b.to_float());                                    \
      auto d = BUILTIN(ai, bi, c.to_float(), idx, CBSZ, ABID);                                     \
      return Value<CFragVec_, S, Cv>(d);                                                           \
    } else {                                                                                       \
      return detail::SOFT_<AVec_, BVec_, S, Cv>(a, b, c, idx);                                     \
    }                                                                                              \
  }

#if !defined(__HIP_DEVICE_COMPILE__) ||                                                            \
    (__has_builtin(__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8) && !defined(__gfx940__) &&        \
     !defined(__gfx941__) && !defined(__gfx942__))
// 16x16x64 (C = v4f).  A=v8 fp8, B=v16 fp8.
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_16x16x64_fp8_fp8, v8e4m3_native, v16e4m3_native,
                        v4f_native, v2i32_smf, v4i32_smf, smfmac_software_16x16x64_fp8,
                        __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8)
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_16x16x64_fp8_bf8, v8e4m3_native, v16e5m2_native,
                        v4f_native, v2i32_smf, v4i32_smf, smfmac_software_16x16x64_fp8,
                        __builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8)
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_16x16x64_bf8_fp8, v8e5m2_native, v16e4m3_native,
                        v4f_native, v2i32_smf, v4i32_smf, smfmac_software_16x16x64_fp8,
                        __builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8)
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_16x16x64_bf8_bf8, v8e5m2_native, v16e5m2_native,
                        v4f_native, v2i32_smf, v4i32_smf, smfmac_software_16x16x64_fp8,
                        __builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8)
// 32x32x32 (C = v16f).  A=v8 fp8, B=v16 fp8.
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_32x32x32_fp8_fp8, v8e4m3_native, v16e4m3_native,
                        v16f_native, v2i32_smf, v4i32_smf, smfmac_software_32x32x32_fp8,
                        __builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8)
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_32x32x32_fp8_bf8, v8e4m3_native, v16e5m2_native,
                        v16f_native, v2i32_smf, v4i32_smf, smfmac_software_32x32x32_fp8,
                        __builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8)
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_32x32x32_bf8_fp8, v8e5m2_native, v16e4m3_native,
                        v16f_native, v2i32_smf, v4i32_smf, smfmac_software_32x32x32_fp8,
                        __builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8)
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_32x32x32_bf8_bf8, v8e5m2_native, v16e5m2_native,
                        v16f_native, v2i32_smf, v4i32_smf, smfmac_software_32x32x32_fp8,
                        __builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8)
#endif

#if !defined(__HIP_DEVICE_COMPILE__) ||                                                            \
    (__has_builtin(__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8) &&                                \
     (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)))
// CDNA3 16x16x64 (C = v4f).  A=v8 AMD-FNUZ fp8, B=v16 AMD-FNUZ fp8.
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_16x16x64_fp8_fp8, v8amd_e4m3_native, v16amd_e4m3_native,
                        v4f_native, v2i32_smf, v4i32_smf, smfmac_software_16x16x64_fp8,
                        __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8)
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_16x16x64_fp8_bf8, v8amd_e4m3_native, v16amd_e5m2_native,
                        v4f_native, v2i32_smf, v4i32_smf, smfmac_software_16x16x64_fp8,
                        __builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8)
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_16x16x64_bf8_fp8, v8amd_e5m2_native, v16amd_e4m3_native,
                        v4f_native, v2i32_smf, v4i32_smf, smfmac_software_16x16x64_fp8,
                        __builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8)
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_16x16x64_bf8_bf8, v8amd_e5m2_native, v16amd_e5m2_native,
                        v4f_native, v2i32_smf, v4i32_smf, smfmac_software_16x16x64_fp8,
                        __builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8)
// CDNA3 32x32x32 (C = v16f).  A=v8 AMD-FNUZ fp8, B=v16 AMD-FNUZ fp8.
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_32x32x32_fp8_fp8, v8amd_e4m3_native, v16amd_e4m3_native,
                        v16f_native, v2i32_smf, v4i32_smf, smfmac_software_32x32x32_fp8,
                        __builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8)
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_32x32x32_fp8_bf8, v8amd_e4m3_native, v16amd_e5m2_native,
                        v16f_native, v2i32_smf, v4i32_smf, smfmac_software_32x32x32_fp8,
                        __builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8)
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_32x32x32_bf8_fp8, v8amd_e5m2_native, v16amd_e4m3_native,
                        v16f_native, v2i32_smf, v4i32_smf, smfmac_software_32x32x32_fp8,
                        __builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8)
FPSAN_DEFINE_SMFMAC_FP8(amdgcn_smfmac_f32_32x32x32_bf8_bf8, v8amd_e5m2_native, v16amd_e5m2_native,
                        v16f_native, v2i32_smf, v4i32_smf, smfmac_software_32x32x32_fp8,
                        __builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8)
#endif

#undef FPSAN_DEFINE_SMFMAC_FP8

// gfx950 fp8-insts SMFMAC (K=128 16x16, K=64 32x32): A = v16 fp8 (=v4i), B =
// v32 fp8 (=v8i).  Native mode forwards to the builtin; FPSan mode runs the
// per-shape software dataflow (smfmac_software_16x16x128_fp8 /
// smfmac_software_32x32x64_fp8) -- both silicon-verified (single-hot probe +
// multi-seed full-random LayoutMatchesHardware).
#define FPSAN_DEFINE_SMFMAC_FP8_BIG(NAME, AVec_, BVec_, CFragVec_, AABI_, BABI_, SOFT_, BUILTIN)   \
  template <int CBSZ = 0, int ABID = 0, Semantics S = Semantics::Native,                           \
            Conversions Cv = Conversions::Explicit>                                                \
  FPSAN_DEVICE Value<CFragVec_, S, Cv> NAME(Value<AVec_, S, Cv> a, Value<BVec_, S, Cv> b,          \
                                            Value<CFragVec_, S, Cv> c, int idx) {                  \
    if constexpr (S == Semantics::Native) {                                                        \
      const AABI_ ai = __builtin_bit_cast(AABI_, a.to_float());                                    \
      const BABI_ bi = __builtin_bit_cast(BABI_, b.to_float());                                    \
      auto d = BUILTIN(ai, bi, c.to_float(), idx, CBSZ, ABID);                                     \
      return Value<CFragVec_, S, Cv>(d);                                                           \
    } else {                                                                                       \
      return detail::SOFT_<AVec_, BVec_, S, Cv>(a, b, c, idx);                                     \
    }                                                                                              \
  }

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8)
FPSAN_DEFINE_SMFMAC_FP8_BIG(amdgcn_smfmac_f32_16x16x128_fp8_fp8, v16e4m3_native, v32e4m3_native,
                            v4f_native, v4i32_smf, v8i32_smf, smfmac_software_16x16x128_fp8,
                            __builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8)
FPSAN_DEFINE_SMFMAC_FP8_BIG(amdgcn_smfmac_f32_16x16x128_fp8_bf8, v16e4m3_native, v32e5m2_native,
                            v4f_native, v4i32_smf, v8i32_smf, smfmac_software_16x16x128_fp8,
                            __builtin_amdgcn_smfmac_f32_16x16x128_fp8_bf8)
FPSAN_DEFINE_SMFMAC_FP8_BIG(amdgcn_smfmac_f32_16x16x128_bf8_fp8, v16e5m2_native, v32e4m3_native,
                            v4f_native, v4i32_smf, v8i32_smf, smfmac_software_16x16x128_fp8,
                            __builtin_amdgcn_smfmac_f32_16x16x128_bf8_fp8)
FPSAN_DEFINE_SMFMAC_FP8_BIG(amdgcn_smfmac_f32_16x16x128_bf8_bf8, v16e5m2_native, v32e5m2_native,
                            v4f_native, v4i32_smf, v8i32_smf, smfmac_software_16x16x128_fp8,
                            __builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8)
FPSAN_DEFINE_SMFMAC_FP8_BIG(amdgcn_smfmac_f32_32x32x64_fp8_fp8, v16e4m3_native, v32e4m3_native,
                            v16f_native, v4i32_smf, v8i32_smf, smfmac_software_32x32x64_fp8,
                            __builtin_amdgcn_smfmac_f32_32x32x64_fp8_fp8)
FPSAN_DEFINE_SMFMAC_FP8_BIG(amdgcn_smfmac_f32_32x32x64_fp8_bf8, v16e4m3_native, v32e5m2_native,
                            v16f_native, v4i32_smf, v8i32_smf, smfmac_software_32x32x64_fp8,
                            __builtin_amdgcn_smfmac_f32_32x32x64_fp8_bf8)
FPSAN_DEFINE_SMFMAC_FP8_BIG(amdgcn_smfmac_f32_32x32x64_bf8_fp8, v16e5m2_native, v32e4m3_native,
                            v16f_native, v4i32_smf, v8i32_smf, smfmac_software_32x32x64_fp8,
                            __builtin_amdgcn_smfmac_f32_32x32x64_bf8_fp8)
FPSAN_DEFINE_SMFMAC_FP8_BIG(amdgcn_smfmac_f32_32x32x64_bf8_bf8, v16e5m2_native, v32e5m2_native,
                            v16f_native, v4i32_smf, v8i32_smf, smfmac_software_32x32x64_fp8,
                            __builtin_amdgcn_smfmac_f32_32x32x64_bf8_bf8)
#endif

#undef FPSAN_DEFINE_SMFMAC_FP8_BIG

} // namespace fpsan

#endif // FPSAN_AMDGCN_SMFMAC_HPP
