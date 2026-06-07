// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/amdgcn_swmmac_gfx12.hpp
// ----------------------------------------------------------------------------
// FPSan wrappers for the gfx12 (RDNA4) sparse WMMA family on wave32:
//
//   swmmac_f32_16x16x32_f16_w32
//   swmmac_f32_16x16x32_bf16_w32
//   swmmac_f16_16x16x32_f16_w32
//   swmmac_f32_16x16x32_fp8_fp8_w32
//   swmmac_f32_16x16x32_fp8_bf8_w32
//   swmmac_f32_16x16x32_bf8_fp8_w32
//   swmmac_f32_16x16x32_bf8_bf8_w32
//
// 2:4 structured sparsity: A is half-density (2 of every 4 K positions
// nonzero) compressed into the same v8 fragment a dense K=16 WMMA uses;
// B holds full density at K=32 (v16); a per-lane 16-bit index selects
// the two live K positions per 4-K group.
//
// Float mode: pass-through to the matching __builtin_amdgcn_swmmac_*.
// This is bit-exact -- the wrapper does no math, just forwards.
//
// FPSan mode: wave-cooperative software dataflow in the payload ring.
// Two shapes share the implementation:
//
//   * f16/bf16 inputs (3 wrappers): per-lane v8 holds 4 K-groups of 2
//     compressed slots each; the 8 groups (k = 4g..4g+3, g in 0..7) are
//     split lane i and lane i+16 by ((g/2)%2). The per-row 16-bit sparse
//     index is split across the same two lanes: 4 nibbles per lane at
//     offsets 4*(g%2) + 8*(g/4).
//   * fp8/bf8 inputs (4 wrappers): per-lane v8 holds the 8 groups packed
//     2 bytes per group; lane split is i and i+16 by (g/4). Index lives
//     4 nibbles per lane at offsets 4*(g%4).
//
// All layouts established with AMD's matrix instruction calculator
// (`v_swmmac_*` register-layout output) and verified bit-for-bit on real
// gfx1201 silicon by the {Bf,F}16,{Fp,Bf}8.LayoutMatchesHardware tests in
// swmmac_gfx12_test.cpp -- the analogous gfx950 SMFMAC harness.
//
// Implementation pattern (same as the gfx950 SMFMAC dataflows in
// amdgcn_smfmac.hpp): ds_bpermute requires the source slot to be uniform
// across the wave, but B's slot index depends on the per-lane sparse K
// selector, so we first prefetch this lane's whole B column (2 source
// lanes hold the 32 K-values for column j; uniform e-loop shuffles fill
// a per-lane Bcol[32]), then index Bcol locally by the per-lane sparse
// k. A's slot only depends on the loop group g, so the A gather stays
// inline. The sparse index nibble is also fetched per group via a
// uniform-slot 16-bit shuffle.
//
// HIP/device-only.  Opt-in (not pulled by <fpsan/fpsan.hpp>).
// ----------------------------------------------------------------------------
#ifndef FPSAN_AMDGCN_SWMMAC_GFX12_HPP
#define FPSAN_AMDGCN_SWMMAC_GFX12_HPP

#include "fpsan/amdgcn_matrix.hpp" // v8h_native, v8bf_native, v8f_native, v8eXm_native, wave_lane, wave_shfl
#include "fpsan/cast.hpp"
#include "fpsan/value.hpp"

#include <cstdint>
#include <type_traits>

#if !defined(__HIP__) && !defined(__CUDACC__)
#    error "fpsan/amdgcn_swmmac_gfx12.hpp is GPU-only; compile as HIP (or CUDA)."
#endif

namespace fpsan
{

    // SWMMAC native fragment vector aliases. B is twice as wide as the dense
    // WMMA 16x16x16 B (because K=32 instead of K=16); A and C/D are the same.
    using v16h_swmmac_native  = _Float16 __attribute__((ext_vector_type(16)));
    using v16bf_swmmac_native = __bf16 __attribute__((ext_vector_type(16)));
    // FP8 SWMMAC A is the existing v8 fp8 fragment (8 bytes per lane = 2 i32);
    // B is twice as wide (16 bytes per lane = 4 i32). Reuse the existing v8
    // alias for A and define a v16-byte fp8 alias for B here.
    namespace detail
    {
        template <class Elem>
        struct v16_byte_fragment
        {
            using bits_t = std::uint8_t __attribute__((ext_vector_type(16)));
            Elem e[16]{};
            FPSAN_HOST_DEVICE constexpr v16_byte_fragment() = default;
            FPSAN_HOST_DEVICE v16_byte_fragment(bits_t b)
            {
                *this = __builtin_bit_cast(v16_byte_fragment, b);
            }
            FPSAN_HOST_DEVICE operator bits_t() const
            {
                return __builtin_bit_cast(bits_t, *this);
            }
            FPSAN_HOST_DEVICE constexpr Elem& operator[](unsigned i)
            {
                return e[i];
            }
            FPSAN_HOST_DEVICE constexpr Elem operator[](unsigned i) const
            {
                return e[i];
            }
            FPSAN_HOST_DEVICE friend auto operator<(const v16_byte_fragment& a,
                                                   const v16_byte_fragment& b)
            {
                using mask_t = std::uint8_t __attribute__((ext_vector_type(16)));
                mask_t m{};
                for(int i = 0; i < 16; ++i)
                    m[i] = a.e[i] < b.e[i] ? std::uint8_t(0xFF) : std::uint8_t(0);
                return m;
            }
        };
    } // namespace detail
    using v16e4m3_swmmac_native = detail::v16_byte_fragment<fp8_e4m3>;
    using v16e5m2_swmmac_native = detail::v16_byte_fragment<fp8_e5m2>;

    // The builtin ABI packs the fp8 fragments into i32 vectors.
    using v2i32_swmmac = int __attribute__((ext_vector_type(2)));
    using v4i32_swmmac = int __attribute__((ext_vector_type(4)));

    namespace detail
    {

        // ---- f16 / bf16 / f16-out SWMMAC 16x16x32 dataflow -----------------------
        // Per-lane fragment layout (verified with AMD matrix calculator + on-silicon):
        //   * A_comp[i][slot]: lane = i + 16*((g/2)%2); slot in v8 = 2*a_gpr + s,
        //                      a_gpr = 2*(g/4) + (g%2); g in 0..7 = K-group of 4
        //                      with k_dense = 4g + p, p = idx-selected (s=0 -> p0,
        //                      s=1 -> p1).
        //   * B[k][j]: lane = j + 16*((k/8)%2); slot in v16 =
        //              8*(k/16) + 2*((k/2)%4) + (k%2).
        //   * D[i][j]: lane = j + 16*(i/8), reg in v8 = i%8 (== dense gfx12 WMMA).
        //   * idx: 16 bits per lane. Compression for (row i, group g) lives at
        //         lane i + 16*((g/2)%2), nibble (g%2 within byte (g/4)); bit_off =
        //         4*(g%2) + 8*(g/4). p0 = nibble&3, p1 = (nibble>>2)&3.
        template <class AVec, class BVec, class CVec, Semantics S, Conversions C>
        FPSAN_DEVICE Value<CVec, S, C> swmmac_software_16x16x32_h(Value<AVec, S, C> a,
                                                                  Value<BVec, S, C> b,
                                                                  Value<CVec, S, C> c,
                                                                  std::uint16_t     idx)
        {
            using DFrag     = Value<CVec, S, C>;
            using AccScalar = typename DFrag::element_type;
            using Acc       = Value<AccScalar, S, C>;
            const int lane  = wave_lane();
            const int j     = lane & 15;

            // Prefetch B column j. Lane j holds k in {0..7, 16..23}; lane j+16 holds
            // k in {8..15, 24..31}. Slot e in 0..15 maps to k = 16*(e/8) + 8*sl + (e%8),
            // for source lane = j + 16*sl. The e-loop is uniform across the wave -- only
            // src varies, which ds_bpermute supports.
            Acc Bcol[32];
            for(int sl = 0; sl < 2; ++sl)
            {
                const int src = j + 16 * sl;
                for(int e = 0; e < 16; ++e)
                {
                    const int k = 16 * (e / 8) + 8 * sl + (e % 8);
                    Bcol[k]     = cast<AccScalar>(wave_shfl(b.get(e), src));
                }
            }

            DFrag     d{};
            const int idx_w = static_cast<int>(static_cast<std::uint32_t>(idx));
            for(int reg = 0; reg < 8; ++reg)
            {
                const int output_i = reg + 8 * (lane >> 4);
                Acc       acc      = c.get(reg);
                for(int g = 0; g < 8; ++g)
                {
                    const int side    = (g >> 1) & 1;
                    const int idxlane = output_i + 16 * side;
                    const int idxA    = __builtin_amdgcn_ds_bpermute(idxlane * 4, idx_w);
                    const int bit_off = 4 * (g & 1) + 8 * (g >> 2);
                    const int nibble  = (idxA >> bit_off) & 0xF;
                    const int p[2]    = {nibble & 3, (nibble >> 2) & 3};
                    const int a_gpr   = 2 * (g >> 2) + (g & 1);
                    const int a_lane  = output_i + 16 * side;
                    for(int s = 0; s < 2; ++s)
                    {
                        const int a_slot = 2 * a_gpr + s;
                        auto      av     = wave_shfl(a.get(a_slot), a_lane);
                        const int k      = 4 * g + p[s];
                        acc              = acc + cast<AccScalar>(av) * Bcol[k];
                    }
                }
                d.set(reg, acc);
            }
            return d;
        }

        // ---- fp8 / bf8 SWMMAC 16x16x32 dataflow ----------------------------------
        // Per-lane fragment layout (verified with AMD matrix calculator):
        //   * A_comp[i][byte]: lane = i + 16*(g/4); byte in v8 = 2*(g%4) + s,
        //                      g in 0..7 (k_dense = 4g + p, p = idx selected).
        //   * B[k][j]: lane = j + 16*(k/16); byte in v16 = k % 16.
        //              (lane j holds k=0..15, lane j+16 holds k=16..31, linear.)
        //   * D[i][j]: lane = j + 16*(i/8), reg in v8f = i%8.
        //   * idx: 16 bits per lane. Compression for (row i, group g) lives at
        //         lane i + 16*(g/4), nibble (g%4); bit_off = 4*(g%4). p0 = nibble&3,
        //         p1 = (nibble>>2)&3.
        // Note the lane split is at K=16 (g/4) for fp8 vs K=8 (g/2)%2 for f16/bf16
        // -- because A's per-lane v8 packs the 8 groups linearly (2 bytes per group),
        // not interleaved-by-K-stride as the f16 case is.
        template <class AVec, class BVec, Semantics S, Conversions C>
        FPSAN_DEVICE Value<v8f_native, S, C> swmmac_software_16x16x32_fp8(Value<AVec, S, C>       a,
                                                                          Value<BVec, S, C>       b,
                                                                          Value<v8f_native, S, C> c,
                                                                          std::uint16_t idx)
        {
            using Acc      = Value<float, S, C>;
            const int lane = wave_lane();
            const int j    = lane & 15;

            // Prefetch B column j (linear byte addressing).
            Acc Bcol[32];
            for(int sl = 0; sl < 2; ++sl)
            {
                const int src = j + 16 * sl;
                for(int e = 0; e < 16; ++e)
                    Bcol[16 * sl + e] = cast<float>(wave_shfl(b.get(e), src));
            }

            Value<v8f_native, S, C> d{};
            const int               idx_w = static_cast<int>(static_cast<std::uint32_t>(idx));
            for(int reg = 0; reg < 8; ++reg)
            {
                const int output_i = reg + 8 * (lane >> 4);
                Acc       acc      = c.get(reg);
                for(int g = 0; g < 8; ++g)
                {
                    const int side    = (g >> 2) & 1;
                    const int idxlane = output_i + 16 * side;
                    const int idxA    = __builtin_amdgcn_ds_bpermute(idxlane * 4, idx_w);
                    const int bit_off = 4 * (g & 3);
                    const int nibble  = (idxA >> bit_off) & 0xF;
                    const int p[2]    = {nibble & 3, (nibble >> 2) & 3};
                    const int a_lane  = output_i + 16 * side;
                    for(int s = 0; s < 2; ++s)
                    {
                        const int byte = 2 * (g & 3) + s;
                        auto      av   = wave_shfl(a.get(byte), a_lane);
                        const int k    = 4 * g + p[s];
                        acc            = acc + cast<float>(av) * Bcol[k];
                    }
                }
                d.set(reg, acc);
            }
            return d;
        }

    } // namespace detail

    // ---- f16 / bf16 / f16-out SWMMAC wrappers ----------------------------------
    // Builtin signatures (per LLVM def, returns first):
    //   swmmac_f32_16x16x32_f16_w32  : (v8h, v16h, v8f, short) -> v8f
    //   swmmac_f32_16x16x32_bf16_w32 : (v8bf, v16bf, v8f, short) -> v8f
    //   swmmac_f16_16x16x32_f16_w32  : (v8h, v16h, v8h, short) -> v8h
#define FPSAN_DEFINE_SWMMAC_GFX12(NAME, AVec_, BVec_, CFragVec_, BUILTIN)                          \
    template <Semantics S, Conversions C>                                                          \
    FPSAN_DEVICE Value<CFragVec_, S, C> NAME(                                                      \
        Value<AVec_, S, C> a, Value<BVec_, S, C> b, Value<CFragVec_, S, C> c, std::uint16_t index) \
    {                                                                                              \
        if constexpr(S == Semantics::Float)                                                        \
        {                                                                                          \
            auto d = BUILTIN(a.to_float(), b.to_float(), c.to_float(), static_cast<short>(index)); \
            return Value<CFragVec_, S, C>(d);                                                      \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            return detail::swmmac_software_16x16x32_h<AVec_, BVec_, CFragVec_, S, C>(              \
                a, b, c, index);                                                                   \
        }                                                                                          \
    }

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32)
    FPSAN_DEFINE_SWMMAC_GFX12(amdgcn_swmmac_f32_16x16x32_f16_w32,
                              v8h_native,
                              v16h_swmmac_native,
                              v8f_native,
                              __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32)
    FPSAN_DEFINE_SWMMAC_GFX12(amdgcn_swmmac_f32_16x16x32_bf16_w32,
                              v8bf_native,
                              v16bf_swmmac_native,
                              v8f_native,
                              __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32)
    FPSAN_DEFINE_SWMMAC_GFX12(amdgcn_swmmac_f16_16x16x32_f16_w32,
                              v8h_native,
                              v16h_swmmac_native,
                              v8h_native,
                              __builtin_amdgcn_swmmac_f16_16x16x32_f16_w32)
#endif

#undef FPSAN_DEFINE_SWMMAC_GFX12

    // ---- FP8 / BF8 SWMMAC wrappers ---------------------------------------------
    // Builtin signature: (v2i32 A, v4i32 B, v8f C, short index) -> v8f
    // A = 8 fp8/bf8 bytes/lane, B = 16 fp8/bf8 bytes/lane, C/D = v8f.
#define FPSAN_DEFINE_SWMMAC_GFX12_FP8(NAME, AVec_, BVec_, BUILTIN)                            \
    template <Semantics S, Conversions C>                                                     \
    FPSAN_DEVICE Value<v8f_native, S, C> NAME(Value<AVec_, S, C>      a,                      \
                                              Value<BVec_, S, C>      b,                      \
                                              Value<v8f_native, S, C> c,                      \
                                              std::uint16_t           index)                  \
    {                                                                                         \
        if constexpr(S == Semantics::Float)                                                   \
        {                                                                                     \
            const v2i32_swmmac ai = __builtin_bit_cast(v2i32_swmmac, a.to_float());           \
            const v4i32_swmmac bi = __builtin_bit_cast(v4i32_swmmac, b.to_float());           \
            auto               d  = BUILTIN(ai, bi, c.to_float(), static_cast<short>(index)); \
            return Value<v8f_native, S, C>(d);                                                \
        }                                                                                     \
        else                                                                                  \
        {                                                                                     \
            return detail::swmmac_software_16x16x32_fp8<AVec_, BVec_, S, C>(a, b, c, index);  \
        }                                                                                     \
    }

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32)
    FPSAN_DEFINE_SWMMAC_GFX12_FP8(amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32,
                                  v8e4m3_native,
                                  v16e4m3_swmmac_native,
                                  __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32)
    FPSAN_DEFINE_SWMMAC_GFX12_FP8(amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32,
                                  v8e4m3_native,
                                  v16e5m2_swmmac_native,
                                  __builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32)
    FPSAN_DEFINE_SWMMAC_GFX12_FP8(amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32,
                                  v8e5m2_native,
                                  v16e4m3_swmmac_native,
                                  __builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32)
    FPSAN_DEFINE_SWMMAC_GFX12_FP8(amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32,
                                  v8e5m2_native,
                                  v16e5m2_swmmac_native,
                                  __builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32)
#endif

#undef FPSAN_DEFINE_SWMMAC_GFX12_FP8

} // namespace fpsan

#endif // FPSAN_AMDGCN_SWMMAC_GFX12_HPP
