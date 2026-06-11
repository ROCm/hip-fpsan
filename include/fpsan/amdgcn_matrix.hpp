// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/amdgcn_matrix.hpp
// ----------------------------------------------------------------------------
// FPSan wrappers for AMDGPU matrix (WMMA/MFMA) intrinsics.
//
// Compiler builtins like __builtin_amdgcn_wmma_* cannot be overloaded or
// redefined (they are reserved and intercepted by Clang). So a ported kernel
// replaces the builtin name with the fpsan:: wrapper, mirroring how it replaces
// `float` with `Value<float,...>`:
//
//     d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
//   becomes
//     d = fpsan::amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
//
// with a, b, c, d being Value fragments (vector Values) instead of raw vectors.
//
//   Semantics::Float : bit-cast to the native vectors and call the real builtin
//                      (true hardware MMA -- fast and bit-faithful).
//   Semantics::FPSan : a wave-cooperative software MMA computing D = A*B + C in
//                      the payload ring, using the hardware fragment layout so
//                      results match an FPSan scalar/Triton reference.
//
// This header is opt-in (not pulled by <fpsan/fpsan.hpp>) and HIP/device only.
// All gfx12 (RDNA4 / gfx1250) 16x16x16 WMMA variants share one fragment layout
// (Wmma16x16x16Layout) and one wave dataflow (wmma_16x16x16_dataflow) -- only
// the element types and the builtin called from Float mode differ. The layout
// was reverse-engineered from AMD's matrix-instruction calculator and
// re-confirmed bit-for-bit on-device by the Wmma.LayoutMatchesHardware test.
// ----------------------------------------------------------------------------
#ifndef FPSAN_AMDGCN_MATRIX_HPP
#define FPSAN_AMDGCN_MATRIX_HPP

#include "fpsan/cast.hpp"
#include "fpsan/detail/config.hpp"
#include "fpsan/detail/fp8.hpp"
#include "fpsan/value.hpp"

#include <cstdint>

#if !defined(__HIP__) && !defined(__CUDACC__)
#error "fpsan/amdgcn_matrix.hpp is GPU-only; compile as HIP (or CUDA)."
#endif

namespace fpsan
{

    // Native vector aliases for the gfx12 16x16x16 WMMA builtin ABI. Per-lane
    // fragment widths: A,B = 8 elements of the input type, C,D = 8 elements of the
    // output type.
    using v8h_native  = _Float16 __attribute__((ext_vector_type(8)));
    using v8f_native  = float __attribute__((ext_vector_type(8)));
    using v8bf_native = __bf16 __attribute__((ext_vector_type(8)));

    namespace detail
    {
        // A POD per-lane N-element fragment of a scalar element type. Clang's
        // ext_vector_type only accepts built-in scalar element types, so for our
        // scalar fp8 types we wrap an array. is_clang_vector_v detects this via
        // operator[], so Value's vector path handles it (bits_type ends up as a
        // uint8 ext_vector of N lanes, with lane arithmetic happening in the payload
        // ring). operator< is provided so the derivation of Value::cmp_t compiles --
        // it returns a per-lane u8 mask; nothing actually uses fp8-vector
        // comparisons today. N is fixed per matrix shape: 8 for the gfx12 fp8 WMMA
        // operand, 16/32 for the SMFMAC / scaled-MFMA operands (aliased in
        // amdgcn_mfma.hpp).
        template <class Elem, int N>
        struct vec_fragment
        {
            using bits_t = std::uint8_t __attribute__((ext_vector_type(N)));
            Elem e[N]{};
            FPSAN_HOST_DEVICE constexpr vec_fragment() = default;
            // Round trip with the raw N-byte payload. Used by Value's ctor / from_bits /
            // to_storage_bits ternaries which need static_cast in both directions.
            FPSAN_HOST_DEVICE vec_fragment(bits_t b)
            {
                *this = __builtin_bit_cast(vec_fragment, b);
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
            FPSAN_HOST_DEVICE friend auto operator<(const vec_fragment& a, const vec_fragment& b)
            {
                using mask_t = std::uint8_t __attribute__((ext_vector_type(N)));
                mask_t m{};
                for(int i = 0; i < N; ++i)
                    m[i] = a.e[i] < b.e[i] ? std::uint8_t(0xFF) : std::uint8_t(0);
                return m;
            }
        };

        // The 8-byte fp8 fragment used by the RDNA4 fp8 WMMAs.
        template <class Elem>
        using v8_fragment = vec_fragment<Elem, 8>;
    } // namespace detail

    // Per-lane fp8 fragment types used by the RDNA4 fp8 WMMAs.
    using v8e4m3_native = detail::v8_fragment<fp8_e4m3>;
    using v8e5m2_native = detail::v8_fragment<fp8_e5m2>;

    namespace detail
    {
        // 4-byte fp8 fragment (4 packed bytes per lane), used by the gfx12
        // 8-bit dot4 family. Same shape as v8_fragment but half the width;
        // Value's vector path treats it as a 4-lane u8.
        template <class Elem>
        struct v4_fragment
        {
            using bits_t = std::uint8_t __attribute__((ext_vector_type(4)));
            Elem e[4]{};
            FPSAN_HOST_DEVICE constexpr v4_fragment() = default;
            FPSAN_HOST_DEVICE v4_fragment(bits_t b)
            {
                *this = __builtin_bit_cast(v4_fragment, b);
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
            FPSAN_HOST_DEVICE friend auto operator<(const v4_fragment& a, const v4_fragment& b)
            {
                using mask_t = std::uint8_t __attribute__((ext_vector_type(4)));
                mask_t m{};
                for(int i = 0; i < 4; ++i)
                    m[i] = a.e[i] < b.e[i] ? std::uint8_t(0xFF) : std::uint8_t(0);
                return m;
            }
        };
    } // namespace detail
    using v4e4m3_native = detail::v4_fragment<fp8_e4m3>;
    using v4e5m2_native = detail::v4_fragment<fp8_e5m2>;
    using v4amd_e4m3_native = detail::v4_fragment<amd_fp8_e4m3>;
    using v4amd_e5m2_native = detail::v4_fragment<amd_fp8_e5m2>;

    // The builtin ABI packs the 8-byte fp8 fragment into a 2-element i32 vector.
    using v2i32_native = int __attribute__((ext_vector_type(2)));

    namespace detail
    {

        // Lane id within the (wave32) wavefront.
        FPSAN_DEVICE inline int wave_lane()
        {
            return __builtin_amdgcn_mbcnt_lo(~0u, 0u);
        }

        // Lane id valid for any wavefront size (32 or 64). mbcnt_hi continues the
        // population count from the low half into the high half of the active mask, so
        // on wave32 (high half empty) it equals wave_lane(), and on wave64 it returns
        // the true 0..63 lane. Use this anywhere a wrapper must be correct on both
        // CDNA (wave64) and RDNA (wave32); pair it with __builtin_amdgcn_wavefrontsize()
        // for the lane count.
        FPSAN_DEVICE inline int wave_lane_full()
        {
            return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
        }

        // Move one scalar Value to this lane from `src_lane` within the wave. The
        // stored representation (payload or float bits) is shuffled verbatim, using the
        // raw cross-lane builtin (no HIP runtime dependency): ds_bpermute addresses the
        // source lane by byte offset (lane*4) and moves a 32-bit word. 64-bit scalars
        // (double) are moved as two halves via two ds_bpermute calls -- correct and
        // portable, without leaning on the HIP-runtime overload of __shfl.
        template <class FT, Semantics S, Conversions C>
        FPSAN_DEVICE Value<FT, S, C> wave_shfl(Value<FT, S, C> v, int src_lane)
        {
            using B = typename Value<FT, S, C>::bits_type;
            static_assert(!Value<FT, S, C>::is_vector, "wave_shfl is scalar-only");
            const auto bits = v.to_storage_bits();
            if constexpr(sizeof(B) <= 4)
            {
                const int w   = static_cast<int>(static_cast<std::uint32_t>(bits));
                const int got = __builtin_amdgcn_ds_bpermute(src_lane * 4, w);
                return Value<FT, S, C>::from_storage_bits(
                    static_cast<B>(static_cast<std::uint32_t>(got)));
            }
            else
            {
                static_assert(sizeof(B) == 8, "wave_shfl supports 1..8 byte scalars");
                const std::uint64_t b64 = static_cast<std::uint64_t>(bits);
                const int           lo  = static_cast<int>(static_cast<std::uint32_t>(b64));
                const int           hi  = static_cast<int>(static_cast<std::uint32_t>(b64 >> 32));
                const int           glo = __builtin_amdgcn_ds_bpermute(src_lane * 4, lo);
                const int           ghi = __builtin_amdgcn_ds_bpermute(src_lane * 4, hi);
                const std::uint64_t g64
                    = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(ghi)) << 32)
                      | static_cast<std::uint64_t>(static_cast<std::uint32_t>(glo));
                return Value<FT, S, C>::from_storage_bits(static_cast<B>(g64));
            }
        }

        // ---- Wave32 16x16x16 WMMA fragment layout (gfx12: RDNA4 + gfx1250) ----------
        // Holds for every gfx12 WMMA whose shape is 16x16x16 wave32, independent of
        // element type: the per-lane v8 vector length scales with element width, the
        // lane/reg mapping does not.
        //   A[m][k]: lane = m + 16*((k>>2)&1), reg = 2*(k>>3)+((k>>1)&1), half = k&1
        //   B[k][n]: lane = n + 16*((k>>2)&1), reg = 2*(k>>3)+((k>>1)&1), half = k&1
        //   C/D[m][n]: lane = n + 16*(m>>3), reg = m&7  (output element index)
        // ab_half() / a 2-half-per-register layout is a property of *16-bit* operand
        // fragments (f16/bf16). For 8-bit operand fragments (fp8) the "half" is a
        // quarter-register; the formulas below still produce the correct 0..7 element
        // index inside the per-lane v8 fragment (= 2*reg + half), since the v8 lane
        // length is independent of element width.
        struct Wmma16x16x16Layout
        {
            FPSAN_DEVICE static int ab_lane(int row_or_col, int k)
            {
                return row_or_col + 16 * ((k >> 2) & 1);
            }
            FPSAN_DEVICE static int ab_reg(int k)
            {
                return 2 * (k >> 3) + ((k >> 1) & 1);
            }
            FPSAN_DEVICE static int ab_half(int k)
            {
                return k & 1;
            }
            // Storage index into the 8-element A/B fragment for (reg, half).
            FPSAN_DEVICE static int ab_index(int k)
            {
                return 2 * ab_reg(k) + ab_half(k);
            }
            // The (m, n) this lane's D/C register e holds.
            FPSAN_DEVICE static int cd_m(int lane, int e)
            {
                return e + 8 * (lane >> 4);
            }
            FPSAN_DEVICE static int cd_n(int lane)
            {
                return lane & 15;
            }
        };

        // Wave-cooperative software MMA shared by every gfx12 16x16x16 WMMA. The
        // fragment layout is fixed; the accumulator type is whatever C and D carry
        // (== AVec/BVec for the f16/bf16 'same-type' variants, == f32 for the f32-out
        // variants). Generic over Value Semantics: real-float arithmetic at
        // Semantics::Float is used as an oracle vs the real builtin in tests, and at
        // Semantics::FPSan the same arithmetic happens in the payload ring.
        template <class AVec, class BVec, class CVec, Semantics S, Conversions C>
        FPSAN_DEVICE Value<CVec, S, C>
            wmma_16x16x16_dataflow(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
        {
            using DFrag     = Value<CVec, S, C>;
            using AccScalar = typename DFrag::element_type;
            using Acc       = Value<AccScalar, S, C>;
            const int lane  = wave_lane();
            const int n     = Wmma16x16x16Layout::cd_n(lane);
            DFrag     d{};
            for(int e = 0; e < 8; ++e)
            {
                const int m   = Wmma16x16x16Layout::cd_m(lane, e);
                Acc       acc = c.get(e); // C[m][n], co-located with D register e
                for(int k = 0; k < 16; ++k)
                {
                    const int idx = Wmma16x16x16Layout::ab_index(k);
                    // Gather A[m][k] and B[k][n] from the lanes that hold them. idx depends
                    // on k only (same on every lane), so the shuffle is well formed.
                    auto av = wave_shfl(a.get(idx), Wmma16x16x16Layout::ab_lane(m, k));
                    auto bv = wave_shfl(b.get(idx), Wmma16x16x16Layout::ab_lane(n, k));
                    acc     = acc + cast<AccScalar>(av) * cast<AccScalar>(bv);
                }
                d.set(e, acc);
            }
            return d;
        }

    } // namespace detail

// =============================================================================
// RDNA4 (gfx1200/gfx1201) wave32 WMMA wrappers. Each wrapper is one
// macro-instantiation: type signature + a Float-mode call to the real builtin +
// FPSan-mode dispatch to the shared software dataflow. AMD's instruction-name
// convention: "fp8" = OCP E4M3FN, "bf8" = OCP E5M2.
//
// Gated on gfx12 (RDNA4): these builtins need the gfx12 wmma-128b-insts +
// wavefrontsize32 features.  On CDNA (gfx9 family, including gfx950) the
// matrix path is MFMA, lives in fpsan/amdgcn_mfma.hpp, and has wave64 ABI.
// =============================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX12__)

// Generic wrapper: passes the native fragment to the builtin as-is. Used for
// f16/bf16 variants where Clang's ext_vector_type IS the builtin ABI.
#define FPSAN_DEFINE_WMMA_16X16X16(NAME, AVec_, BVec_, CVec_, BUILTIN)    \
    template <Semantics S, Conversions C>                                 \
    FPSAN_DEVICE Value<CVec_, S, C> NAME(                                 \
        Value<AVec_, S, C> a, Value<BVec_, S, C> b, Value<CVec_, S, C> c) \
    {                                                                     \
        if constexpr(S == Semantics::Float)                               \
        {                                                                 \
            CVec_ d = BUILTIN(a.to_float(), b.to_float(), c.to_float());  \
            return Value<CVec_, S, C>(d);                                 \
        }                                                                 \
        else                                                              \
        {                                                                 \
            return detail::wmma_16x16x16_dataflow(a, b, c);               \
        }                                                                 \
    }

// Fp8 wrapper: bit-casts the 8-byte v8_fragment to the v2i32 ABI the builtin
// expects (Clang ext_vector_type doesn't accept fp8 element types).
#define FPSAN_DEFINE_WMMA_16X16X16_FP8(NAME, AVec_, BVec_, BUILTIN)                \
    template <Semantics S, Conversions C>                                          \
    FPSAN_DEVICE Value<v8f_native, S, C> NAME(                                     \
        Value<AVec_, S, C> a, Value<BVec_, S, C> b, Value<v8f_native, S, C> c)     \
    {                                                                              \
        if constexpr(S == Semantics::Float)                                        \
        {                                                                          \
            v8f_native d = BUILTIN(__builtin_bit_cast(v2i32_native, a.to_float()), \
                                   __builtin_bit_cast(v2i32_native, b.to_float()), \
                                   c.to_float());                                  \
            return Value<v8f_native, S, C>(d);                                     \
        }                                                                          \
        else                                                                       \
        {                                                                          \
            return detail::wmma_16x16x16_dataflow(a, b, c);                        \
        }                                                                          \
    }

    FPSAN_DEFINE_WMMA_16X16X16(amdgcn_wmma_f32_16x16x16_f16_w32,
                               v8h_native,
                               v8h_native,
                               v8f_native,
                               __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
    FPSAN_DEFINE_WMMA_16X16X16(amdgcn_wmma_f16_16x16x16_f16_w32,
                               v8h_native,
                               v8h_native,
                               v8h_native,
                               __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12)
    FPSAN_DEFINE_WMMA_16X16X16(amdgcn_wmma_f32_16x16x16_bf16_w32,
                               v8bf_native,
                               v8bf_native,
                               v8f_native,
                               __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12)
    FPSAN_DEFINE_WMMA_16X16X16(amdgcn_wmma_bf16_16x16x16_bf16_w32,
                               v8bf_native,
                               v8bf_native,
                               v8bf_native,
                               __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12)

    FPSAN_DEFINE_WMMA_16X16X16_FP8(amdgcn_wmma_f32_16x16x16_fp8_fp8_w32,
                                   v8e4m3_native,
                                   v8e4m3_native,
                                   __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12)
    FPSAN_DEFINE_WMMA_16X16X16_FP8(amdgcn_wmma_f32_16x16x16_fp8_bf8_w32,
                                   v8e4m3_native,
                                   v8e5m2_native,
                                   __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12)
    FPSAN_DEFINE_WMMA_16X16X16_FP8(amdgcn_wmma_f32_16x16x16_bf8_fp8_w32,
                                   v8e5m2_native,
                                   v8e4m3_native,
                                   __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12)
    FPSAN_DEFINE_WMMA_16X16X16_FP8(amdgcn_wmma_f32_16x16x16_bf8_bf8_w32,
                                   v8e5m2_native,
                                   v8e5m2_native,
                                   __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12)

#undef FPSAN_DEFINE_WMMA_16X16X16
#undef FPSAN_DEFINE_WMMA_16X16X16_FP8

#endif // defined(__GFX12__)

} // namespace fpsan

#endif // FPSAN_AMDGCN_MATRIX_HPP
