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
//   Semantics::Native : bit-cast to the native vectors and call the real builtin
//                      (true hardware MMA -- fast and bit-faithful).
//   Semantics::Triton : a wave-cooperative software MMA computing D = A*B + C in
//                      the payload ring, using the hardware fragment layout so
//                      results match an FPSan scalar/Triton reference.
//
// This header is opt-in (not pulled by <fpsan/fpsan.hpp>) and HIP/device only.
// gfx11 (RDNA3), gfx12 (RDNA4), and gfx1250 expose related WMMA builtins,
// but their A/B fragment ABI, C/D layout, and supported K-shapes differ. Keep
// the layouts separate even where wrapper names overlap. The layouts are taken
// from AMD's matrix-instruction calculator and re-confirmed on-device by the
// Wmma*.LayoutMatchesHardware tests. gfx1250's distinct K=32/64/128 family
// remains gated on its own builtins below.
// ----------------------------------------------------------------------------
#ifndef FPSAN_AMDGCN_MATRIX_HPP
#define FPSAN_AMDGCN_MATRIX_HPP

#include "fpsan/cast.hpp"
#include "fpsan/detail/config.hpp"
#include "fpsan/detail/fp8.hpp"
#include "fpsan/detail/subbyte_widen.hpp"
#include "fpsan/math.hpp" // for fpsan::max (WMMA C-modifier abs)
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
    using v4h_native  = _Float16 __attribute__((ext_vector_type(4)));
    using v4f_native  = float __attribute__((ext_vector_type(4)));
    using v4bf_native = __bf16 __attribute__((ext_vector_type(4)));
    using v8h_native  = _Float16 __attribute__((ext_vector_type(8)));
    using v8f_native  = float __attribute__((ext_vector_type(8)));
    using v8bf_native = __bf16 __attribute__((ext_vector_type(8)));
    // gfx1250 32x16x128 f4 accumulator: 16 f32 per lane.
    using v16f_native = float __attribute__((ext_vector_type(16)));

    // gfx1250 K=4 f32 operand fragments hold 2 elements per lane (A is 16x4, B is
    // 4x16, wave32 => 64/32 = 2 per lane); C/D stay v8.
    using v2f_native = float __attribute__((ext_vector_type(2)));

    // gfx1250 K=32 16-bit operand fragments hold 16 elements per lane (A is
    // 16x32, B is 32x16, wave32 => 512/32 = 16 per lane); C/D stay v8.
    using v16h_native  = _Float16 __attribute__((ext_vector_type(16)));
    using v16bf_native = __bf16 __attribute__((ext_vector_type(16)));

    // gfx1250 SWMMAC 16x16x64 16-bit fragments: the dense B operand holds 32
    // elements per lane (B is 64x16, wave32 => 1024/32 = 32 per lane); C/D stay v8.
    using v32h_native  = _Float16 __attribute__((ext_vector_type(32)));
    using v32bf_native = __bf16 __attribute__((ext_vector_type(32)));

    // Native vector aliases for the gfx11 16x16x16 WMMA builtin ABI. A/B are
    // full 256-bit replicated operands. f32 accumulators use v8f in wave32 and
    // v4f in wave64; f16/bf16 accumulator/output variants use 16 half-slots in
    // wave32 and 8 half-slots in wave64, with OPSEL selecting either the low or
    // high half of each 32-bit accumulator register.
    using v16h_wmma_native   = _Float16 __attribute__((ext_vector_type(16)));
    using v8i16_wmma_native  = short __attribute__((ext_vector_type(8)));
    using v16bf_wmma_native  = __bf16 __attribute__((ext_vector_type(16)));
    using v16i16_wmma_native = short __attribute__((ext_vector_type(16)));

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

        // The 32-byte fp8 fragment used by the gfx1250 K=64 fp8 WMMAs (32 fp8 codes
        // per lane = 16x64 / 32 lanes). Bit-castable to the v8i32 builtin ABI.
        template <class Elem>
        using v32_fragment = vec_fragment<Elem, 32>;

        // The 64-byte fp8 fragment used by the gfx1250 K=128 fp8 WMMAs (64 fp8 codes
        // per lane = 16x128 / 32 lanes). Bit-castable to the v16i32 builtin ABI.
        template <class Elem>
        using v64_fragment = vec_fragment<Elem, 64>;

        // ---- WMMA C-accumulator modifier (gfx1250) ---------------------------------
        // Every gfx1250 FP WMMA takes an i16 immediate that modifies the C
        // accumulator before it is added: D = A*B + mod(C). Silicon-grounded (probed
        // across all shapes): only the low 2 bits are live -- bit0 = negate,
        // bit1 = abs -- applied abs-then-neg, so
        //   0: C   1: -C   2: |C|   3: -|C|   (higher bits ignored).
        // Operand negation (the leading neg(A)/neg(B) bools) is reserved-zero [0,0]
        // on every FP shape, and matrix_a/b_reuse are perf hints with no numeric
        // effect -- so this is the only numerically-meaningful WMMA modifier.
        // Float mode lets the builtin apply it (we pass the constant); FPSan mode
        // pre-applies it to the C fragment here, leaving the dataflows unchanged.
        // abs is computed as max(c, -c) so the payload matches the host reference.
        template <int Cmod, class CV, Semantics S, Conversions C>
        FPSAN_DEVICE Value<CV, S, C> wmma_apply_cmod(Value<CV, S, C> c)
        {
            static_assert((Cmod & ~0x3) == 0, "WMMA C-modifier is 2 bits: bit0 = neg, bit1 = abs");
            if constexpr((Cmod & 0x3) == 0)
            {
                return c;
            }
            else
            {
                Value<CV, S, C> r{};
                for(unsigned e = 0; e < Value<CV, S, C>::lanes; ++e)
                {
                    auto v = c.get(e);
                    if constexpr(Cmod & 0x2)
                        v = fpsan::max(v, -v); // abs
                    if constexpr(Cmod & 0x1)
                        v = -v; // neg
                    r.set(e, v);
                }
                return r;
            }
        }
    } // namespace detail

    // Per-lane fp8 fragment types used by the RDNA4 fp8 WMMAs.
    using v8e4m3_native = detail::v8_fragment<fp8_e4m3>;
    using v8e5m2_native = detail::v8_fragment<fp8_e5m2>;

    // Per-lane fp8 fragment types used by the gfx1250 K=64 fp8 WMMAs.
    using v32e4m3_native = detail::v32_fragment<fp8_e4m3>;
    using v32e5m2_native = detail::v32_fragment<fp8_e5m2>;

    // Per-lane fp8 fragment types used by the gfx1250 K=128 fp8 WMMAs.
    using v64e4m3_native = detail::v64_fragment<fp8_e4m3>;
    using v64e5m2_native = detail::v64_fragment<fp8_e5m2>;

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
    using v4e4m3_native     = detail::v4_fragment<fp8_e4m3>;
    using v4e5m2_native     = detail::v4_fragment<fp8_e5m2>;
    using v4amd_e4m3_native = detail::v4_fragment<amd_fp8_e4m3>;
    using v4amd_e5m2_native = detail::v4_fragment<amd_fp8_e5m2>;

    // The builtin ABI packs the 8-byte fp8 fragment into a 2-element i32 vector.
    using v4i16_native = short __attribute__((ext_vector_type(4)));
    using v2i32_native = int __attribute__((ext_vector_type(2)));
    // ... and the 32-byte K=64 fp8 fragment into an 8-element i32 vector.
    using v8i32_native = int __attribute__((ext_vector_type(8)));
    // ... and the 64-byte K=128 fp8 fragment into a 16-element i32 vector.
    using v16i32_native = int __attribute__((ext_vector_type(16)));

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

        FPSAN_DEVICE inline std::uint32_t wave_shfl_word(std::uint32_t word, int src_lane)
        {
// RDNA3 wave64 DS permute lane selection is modulo 32. When a gfx11
// lane requests the opposite half, swap halves first, then do the
// same-half DS gather by the low 5 bits of the absolute source lane.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__) \
    && __has_builtin(__builtin_amdgcn_permlane64)
            if(__builtin_amdgcn_wavefrontsize() == 64)
            {
                const int lane = wave_lane_full();
                if(((lane ^ src_lane) & 32) != 0)
                {
                    word = static_cast<std::uint32_t>(
                        __builtin_amdgcn_permlane64(static_cast<int>(word)));
                }
                src_lane &= 31;
            }
#endif
            return static_cast<std::uint32_t>(
                __builtin_amdgcn_ds_bpermute(src_lane * 4, static_cast<int>(word)));
        }

        // Move one scalar Value to this lane from `src_lane` within the wave. The
        // stored representation (payload or float bits) is shuffled verbatim, using
        // raw cross-lane builtins (no HIP runtime dependency). 64-bit scalars
        // (double) are moved as two halves via two 32-bit shuffles.
        template <class FT, Semantics S, Conversions C>
        FPSAN_DEVICE Value<FT, S, C> wave_shfl(Value<FT, S, C> v, int src_lane)
        {
            using B = typename Value<FT, S, C>::bits_type;
            static_assert(!Value<FT, S, C>::is_vector, "wave_shfl is scalar-only");
            const auto bits = v.to_storage_bits();
            if constexpr(sizeof(B) <= 4)
            {
                const auto got = wave_shfl_word(static_cast<std::uint32_t>(bits), src_lane);
                return Value<FT, S, C>::from_storage_bits(static_cast<B>(got));
            }
            else
            {
                static_assert(sizeof(B) == 8, "wave_shfl supports 1..8 byte scalars");
                const std::uint64_t b64 = static_cast<std::uint64_t>(bits);
                const auto          glo = wave_shfl_word(static_cast<std::uint32_t>(b64), src_lane);
                const auto ghi = wave_shfl_word(static_cast<std::uint32_t>(b64 >> 32), src_lane);
                const std::uint64_t g64
                    = (static_cast<std::uint64_t>(ghi) << 32) | static_cast<std::uint64_t>(glo);
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
                const int lane = row_or_col + 16 * ((k >> 2) & 1);
                return __builtin_amdgcn_wavefrontsize() == 64 ? lane + 32 * ((k >> 3) & 1) : lane;
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
                if(__builtin_amdgcn_wavefrontsize() == 64)
                    return 2 * ((k >> 1) & 1) + ab_half(k);
                return 2 * ab_reg(k) + ab_half(k);
            }
            // The (m, n) this lane's D/C register e holds.
            FPSAN_DEVICE static int cd_m(int lane, int e)
            {
                if(__builtin_amdgcn_wavefrontsize() == 64)
                    return e + 8 * ((lane >> 4) & 1) + 4 * (lane >> 5);
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
        // Semantics::Native is used as an oracle vs the real builtin in tests, and at
        // Semantics::Triton the same arithmetic happens in the payload ring.
        template <class AVec, class BVec, class CVec, Semantics S, Conversions C>
        FPSAN_DEVICE Value<CVec, S, C>
            wmma_16x16x16_dataflow(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
        {
            using DFrag     = Value<CVec, S, C>;
            using AccScalar = typename DFrag::element_type;
            using Acc       = Value<AccScalar, S, C>;
            const int lane  = wave_lane_full();
            const int n     = Wmma16x16x16Layout::cd_n(lane);
            DFrag     d{};
            for(int e = 0; e < static_cast<int>(DFrag::lanes); ++e)
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

        // ---- 16x16x16 WMMA fragment layout (gfx11: RDNA3) ----------------------
        // Authoritative formulas from AMD's matrix-instruction calculator:
        //   A[m][k]: lane = m + 16*q for every q in the wave, reg = floor(k/2),
        //             half = k&1
        //   B[k][n]: lane = n + 16*q for every q in the wave, reg = floor(k/2),
        //             half = k&1
        //   C/D f32[m][n]: lane = (16*m + n) % wave_size,
        //                   reg = floor(m / (wave_size/16))
        //   C/D f16/bf16[m][n]: same lane,
        //                       reg = 2*floor(m / (wave_size/16))+OPSEL
        // The v16 A/B vector index is exactly k. For f16/bf16 outputs, vector
        // indices 2*r+OPSEL are defined; the opposite half is undefined by the
        // non-tied hardware builtins.
        struct WmmaGfx11_16x16x16Layout
        {
            FPSAN_DEVICE static int ab_lane(int row_or_col)
            {
                return row_or_col;
            }
            FPSAN_DEVICE static int ab_index(int k)
            {
                return k;
            }
            FPSAN_DEVICE static int cd_n(int lane)
            {
                return lane & 15;
            }
            FPSAN_DEVICE static int cd_m(int lane, int reg, int wave_size)
            {
                return (wave_size / 16) * reg + (lane >> 4);
            }
            FPSAN_DEVICE static int cd_half_index(int reg, bool opsel)
            {
                return 2 * reg + (opsel ? 1 : 0);
            }
        };

        // gfx11 f32-output WMMA: A/B are v16 replicated operands, C/D are v8f
        // in wave32 and v4f in wave64.
        template <class AVec, class BVec, class CVec, Semantics S, Conversions C>
        FPSAN_DEVICE Value<CVec, S, C> wmma_gfx11_f32_16x16x16_dataflow(Value<AVec, S, C> a,
                                                                        Value<BVec, S, C> b,
                                                                        Value<CVec, S, C> c)
        {
            using DFrag            = Value<CVec, S, C>;
            using Acc              = Value<float, S, C>;
            static constexpr int E = static_cast<int>(DFrag::lanes);
            static_assert(E == 4 || E == 8, "gfx11 f32 WMMA C/D must be v4f or v8f");
            const int lane      = wave_lane_full();
            const int wave_size = __builtin_amdgcn_wavefrontsize();
            const int n         = WmmaGfx11_16x16x16Layout::cd_n(lane);
            DFrag     d{};
            for(int e = 0; e < E; ++e)
            {
                const int m   = WmmaGfx11_16x16x16Layout::cd_m(lane, e, wave_size);
                Acc       acc = c.get(e);
                for(int k = 0; k < 16; ++k)
                {
                    const int idx = WmmaGfx11_16x16x16Layout::ab_index(k);
                    auto      av  = wave_shfl(a.get(idx), WmmaGfx11_16x16x16Layout::ab_lane(m));
                    auto      bv  = wave_shfl(b.get(idx), WmmaGfx11_16x16x16Layout::ab_lane(n));
                    acc           = acc + cast<float>(av) * cast<float>(bv);
                }
                d.set(e, acc);
            }
            return d;
        }

        // gfx11 f16/bf16-output WMMA: OPSEL selects the defined half of each
        // output register. The other half is undefined by the non-tied hardware
        // builtins, so the FPSan path only promises the selected half.
        template <bool Opsel,
                  bool PreserveUnselected = false,
                  class AVec,
                  class BVec,
                  class CVec,
                  Semantics   S,
                  Conversions C>
        FPSAN_DEVICE Value<CVec, S, C> wmma_gfx11_half_16x16x16_dataflow(Value<AVec, S, C> a,
                                                                         Value<BVec, S, C> b,
                                                                         Value<CVec, S, C> c)
        {
            using DFrag            = Value<CVec, S, C>;
            using AccScalar        = typename DFrag::element_type;
            using Acc              = Value<AccScalar, S, C>;
            static constexpr int E = static_cast<int>(DFrag::lanes);
            static_assert(E == 8 || E == 16, "gfx11 f16/bf16 WMMA C/D must be v8 or v16");
            const int lane      = wave_lane_full();
            const int wave_size = __builtin_amdgcn_wavefrontsize();
            const int n         = WmmaGfx11_16x16x16Layout::cd_n(lane);
            DFrag     d{};
            if constexpr(PreserveUnselected)
                d = c;
            for(int r = 0; r < E / 2; ++r)
            {
                const int out_idx = WmmaGfx11_16x16x16Layout::cd_half_index(r, Opsel);
                const int m       = WmmaGfx11_16x16x16Layout::cd_m(lane, r, wave_size);
                Acc       acc     = c.get(out_idx);
                for(int k = 0; k < 16; ++k)
                {
                    const int idx = WmmaGfx11_16x16x16Layout::ab_index(k);
                    auto      av  = wave_shfl(a.get(idx), WmmaGfx11_16x16x16Layout::ab_lane(m));
                    auto      bv  = wave_shfl(b.get(idx), WmmaGfx11_16x16x16Layout::ab_lane(n));
                    acc           = acc + cast<AccScalar>(av) * cast<AccScalar>(bv);
                }
                d.set(out_idx, acc);
            }
            return d;
        }

        // ---- Wave32 16x16x4 WMMA fragment layout (gfx1250, f32 operands) ------------
        // The gfx1250 K=4 f32 shape (wmma_f32_16x16x4_f32): A is 16x4, B is 4x16,
        // both f32, so each lane holds 2 operand elements (16*4/32 = 2), v2f ABI.
        // C/D is the same 16x16 wave32 accumulator as every other gfx1250 WMMA.
        // The A/B layout is the same family as K=32/64/128, but with only two K bits
        // there is no bit2 to select the lane half, so the half-select moves to the
        // next available bit (bit1); the low bit (k&1) is the per-lane v2 sub-index:
        //   index = k & 1                     (0..1)
        //   lane  = row + 16 * ((k >> 1) & 1)
        // i.e. lanes 0..15 hold A[row][0..1], lanes 16..31 hold A[row][2..3].
        // Validated end-to-end by the self-checking layout test against the builtin.
        struct Wmma16x16x4Layout
        {
            FPSAN_DEVICE static int ab_lane(int row_or_col, int k)
            {
                return row_or_col + 16 * ((k >> 1) & 1);
            }
            FPSAN_DEVICE static int ab_index(int k)
            {
                return k & 1;
            }
            FPSAN_DEVICE static int cd_m(int lane, int e)
            {
                return e + 8 * (lane >> 4);
            }
            FPSAN_DEVICE static int cd_n(int lane)
            {
                return lane & 15;
            }
        };

        // Wave-cooperative software MMA for the gfx1250 16x16x4 f32 shape (K=4
        // contraction, v2f A/B fragments, v8f C/D accumulator). Generic over
        // Semantics so Float mode acts as a host/builtin oracle and FPSan mode runs
        // the same arithmetic in the payload ring.
        template <class AVec, class BVec, class CVec, Semantics S, Conversions C>
        FPSAN_DEVICE Value<CVec, S, C>
            wmma_16x16x4_dataflow(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
        {
            using DFrag     = Value<CVec, S, C>;
            using AccScalar = typename DFrag::element_type;
            using Acc       = Value<AccScalar, S, C>;
            const int lane  = wave_lane();
            const int n     = Wmma16x16x4Layout::cd_n(lane);
            DFrag     d{};
            for(int e = 0; e < 8; ++e)
            {
                const int m   = Wmma16x16x4Layout::cd_m(lane, e);
                Acc       acc = c.get(e);
                for(int k = 0; k < 4; ++k)
                {
                    const int idx = Wmma16x16x4Layout::ab_index(k);
                    auto      av  = wave_shfl(a.get(idx), Wmma16x16x4Layout::ab_lane(m, k));
                    auto      bv  = wave_shfl(b.get(idx), Wmma16x16x4Layout::ab_lane(n, k));
                    acc           = acc + cast<AccScalar>(av) * cast<AccScalar>(bv);
                }
                d.set(e, acc);
            }
            return d;
        }

        // ---- Wave32 16x16x32 WMMA fragment layout (gfx1250) -------------------------
        // The gfx1250 K=32 16-bit-operand shapes (f16/bf16). C/D is the same 16x16
        // wave32 accumulator layout as 16x16x16 (cd_m/cd_n reused below via the K=16
        // struct). The A/B operand layout is the gfx12 K=16 mapping
        // extended by the one extra K bit: bit2 of k still selects the lane half
        // (row vs row+16); bits {0,1,3,4} pack the 16-element (v16) per-lane index.
        //   half = k&1; lane = row + 16*((k>>2)&1)
        //   reg  = ((k>>1)&1) + 2*((k>>3)&1) + 4*((k>>4)&1)   (0..7)
        //   index= 2*reg + half                                (0..15)
        // This is validated end-to-end by the self-checking test: the Float path
        // (real builtin) and the FPSan path (this gather) must both match an
        // independent host matmul over identically-staged fragments, which only
        // holds if this layout equals the hardware fragment ABI.
        struct Wmma16x16x32Layout
        {
            FPSAN_DEVICE static int ab_lane(int row_or_col, int k)
            {
                return row_or_col + 16 * ((k >> 2) & 1);
            }
            FPSAN_DEVICE static int ab_index(int k)
            {
                const int reg = ((k >> 1) & 1) + 2 * ((k >> 3) & 1) + 4 * ((k >> 4) & 1);
                return 2 * reg + (k & 1);
            }
            FPSAN_DEVICE static int cd_m(int lane, int e)
            {
                return e + 8 * (lane >> 4);
            }
            FPSAN_DEVICE static int cd_n(int lane)
            {
                return lane & 15;
            }
        };

        // Wave-cooperative software MMA for the gfx1250 16x16x32 shapes. Identical in
        // shape to the 16x16x16 dataflow but with a K=32 contraction and v16 A/B
        // fragments. Generic over Semantics so Float mode acts as a host/builtin
        // oracle and FPSan mode runs the same arithmetic in the payload ring.
        template <class AVec, class BVec, class CVec, Semantics S, Conversions C>
        FPSAN_DEVICE Value<CVec, S, C>
            wmma_16x16x32_dataflow(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
        {
            using DFrag     = Value<CVec, S, C>;
            using AccScalar = typename DFrag::element_type;
            using Acc       = Value<AccScalar, S, C>;
            const int lane  = wave_lane();
            const int n     = Wmma16x16x32Layout::cd_n(lane);
            DFrag     d{};
            for(int e = 0; e < 8; ++e)
            {
                const int m   = Wmma16x16x32Layout::cd_m(lane, e);
                Acc       acc = c.get(e);
                for(int k = 0; k < 32; ++k)
                {
                    const int idx = Wmma16x16x32Layout::ab_index(k);
                    auto      av  = wave_shfl(a.get(idx), Wmma16x16x32Layout::ab_lane(m, k));
                    auto      bv  = wave_shfl(b.get(idx), Wmma16x16x32Layout::ab_lane(n, k));
                    acc           = acc + cast<AccScalar>(av) * cast<AccScalar>(bv);
                }
                d.set(e, acc);
            }
            return d;
        }

        // Mixed accumulator/output variant of the K=32 dataflow: accumulate in the
        // C fragment's element type (e.g. f32) over the full K=32 contraction, then
        // round the final sum to the D fragment's element type (e.g. bf16). Used by
        // wmma_bf16f32_16x16x32_bf16 (f32 accumulate -> bf16 output); the single
        // final cast matches the hardware's "f32 accumulate, narrow once" semantics.
        template <class DVec, class AVec, class BVec, class CVec, Semantics S, Conversions C>
        FPSAN_DEVICE Value<DVec, S, C> wmma_16x16x32_dataflow_cdiff(Value<AVec, S, C> a,
                                                                    Value<BVec, S, C> b,
                                                                    Value<CVec, S, C> c)
        {
            using AccScalar        = typename Value<CVec, S, C>::element_type;
            using DScalar          = typename Value<DVec, S, C>::element_type;
            using Acc              = Value<AccScalar, S, C>;
            const int         lane = wave_lane();
            const int         n    = Wmma16x16x32Layout::cd_n(lane);
            Value<DVec, S, C> d{};
            for(int e = 0; e < 8; ++e)
            {
                const int m   = Wmma16x16x32Layout::cd_m(lane, e);
                Acc       acc = c.get(e);
                for(int k = 0; k < 32; ++k)
                {
                    const int idx = Wmma16x16x32Layout::ab_index(k);
                    auto      av  = wave_shfl(a.get(idx), Wmma16x16x32Layout::ab_lane(m, k));
                    auto      bv  = wave_shfl(b.get(idx), Wmma16x16x32Layout::ab_lane(n, k));
                    acc           = acc + cast<AccScalar>(av) * cast<AccScalar>(bv);
                }
                d.set(e, cast<DScalar>(acc));
            }
            return d;
        }

        // ---- Wave32 16x16x64 WMMA fragment layout (gfx1250, 8-bit operands) ---------
        // K=64 fp8/bf8 operands: 32 codes per lane (16x64 / 32 lanes), v8i32 ABI.
        // C/D is the same 16x16 wave32 accumulator. The A/B operand layout continues
        // the same family as K=16/K=32: bit2 of k selects the lane half; the other k
        // bits {0,1,3,4,5} pack the 32-element fragment index.
        //   half = k&1; lane = row + 16*((k>>2)&1)
        //   reg  = ((k>>1)&1) + 2*((k>>3)&1) + 4*((k>>4)&1) + 8*((k>>5)&1)   (0..15)
        //   index= 2*reg + half                                              (0..31)
        // Validated end-to-end by the self-checking layout test against the builtin.
        struct Wmma16x16x64Layout
        {
            FPSAN_DEVICE static int ab_lane(int row_or_col, int k)
            {
                return row_or_col + 16 * ((k >> 2) & 1);
            }
            FPSAN_DEVICE static int ab_index(int k)
            {
                const int reg
                    = ((k >> 1) & 1) + 2 * ((k >> 3) & 1) + 4 * ((k >> 4) & 1) + 8 * ((k >> 5) & 1);
                return 2 * reg + (k & 1);
            }
            FPSAN_DEVICE static int cd_m(int lane, int e)
            {
                return e + 8 * (lane >> 4);
            }
            FPSAN_DEVICE static int cd_n(int lane)
            {
                return lane & 15;
            }
        };

        // Wave-cooperative software MMA for the gfx1250 16x16x64 fp8/bf8 shapes
        // (K=64 contraction, v32 fp8 A/B fragments, v8 C/D accumulator). Same-type
        // C/D (f32->f32 or f16->f16). Generic over Semantics.
        template <class AVec, class BVec, class CVec, Semantics S, Conversions C>
        FPSAN_DEVICE Value<CVec, S, C>
            wmma_16x16x64_dataflow(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
        {
            using DFrag     = Value<CVec, S, C>;
            using AccScalar = typename DFrag::element_type;
            using Acc       = Value<AccScalar, S, C>;
            const int lane  = wave_lane();
            const int n     = Wmma16x16x64Layout::cd_n(lane);
            DFrag     d{};
            for(int e = 0; e < 8; ++e)
            {
                const int m   = Wmma16x16x64Layout::cd_m(lane, e);
                Acc       acc = c.get(e);
                for(int k = 0; k < 64; ++k)
                {
                    const int idx = Wmma16x16x64Layout::ab_index(k);
                    auto      av  = wave_shfl(a.get(idx), Wmma16x16x64Layout::ab_lane(m, k));
                    auto      bv  = wave_shfl(b.get(idx), Wmma16x16x64Layout::ab_lane(n, k));
                    acc           = acc + cast<AccScalar>(av) * cast<AccScalar>(bv);
                }
                d.set(e, acc);
            }
            return d;
        }

        // ---- Wave32 16x16x128 WMMA fragment layout (gfx1250, 8-bit operands) --------
        // K=128 fp8/bf8 operands: 64 codes per lane (16x128 / 32 lanes), v16i32 ABI.
        // Continues the same family: bit2 of k selects the lane half; k bits
        // {0,1,3,4,5,6} pack the 64-element fragment index.
        //   half = k&1; lane = row + 16*((k>>2)&1)
        //   reg  = sum_{b in {1,3,4,5,6}} bit_b(k) * 2^rank   (0..31)
        //   index= 2*reg + half                               (0..63)
        struct Wmma16x16x128Layout
        {
            FPSAN_DEVICE static int ab_lane(int row_or_col, int k)
            {
                return row_or_col + 16 * ((k >> 2) & 1);
            }
            FPSAN_DEVICE static int ab_index(int k)
            {
                const int reg = ((k >> 1) & 1) + 2 * ((k >> 3) & 1) + 4 * ((k >> 4) & 1)
                                + 8 * ((k >> 5) & 1) + 16 * ((k >> 6) & 1);
                return 2 * reg + (k & 1);
            }
            FPSAN_DEVICE static int cd_m(int lane, int e)
            {
                return e + 8 * (lane >> 4);
            }
            FPSAN_DEVICE static int cd_n(int lane)
            {
                return lane & 15;
            }
        };

        // Wave-cooperative software MMA for the gfx1250 16x16x128 fp8/bf8 shapes
        // (K=128 contraction, v64 fp8 A/B fragments, v8 C/D accumulator).
        template <class AVec, class BVec, class CVec, Semantics S, Conversions C>
        FPSAN_DEVICE Value<CVec, S, C>
            wmma_16x16x128_dataflow(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
        {
            using DFrag     = Value<CVec, S, C>;
            using AccScalar = typename DFrag::element_type;
            using Acc       = Value<AccScalar, S, C>;
            const int lane  = wave_lane();
            const int n     = Wmma16x16x128Layout::cd_n(lane);
            DFrag     d{};
            for(int e = 0; e < 8; ++e)
            {
                const int m   = Wmma16x16x128Layout::cd_m(lane, e);
                Acc       acc = c.get(e);
                for(int k = 0; k < 128; ++k)
                {
                    const int idx = Wmma16x16x128Layout::ab_index(k);
                    auto      av  = wave_shfl(a.get(idx), Wmma16x16x128Layout::ab_lane(m, k));
                    auto      bv  = wave_shfl(b.get(idx), Wmma16x16x128Layout::ab_lane(n, k));
                    acc           = acc + cast<AccScalar>(av) * cast<AccScalar>(bv);
                }
                d.set(e, acc);
            }
            return d;
        }

        // ---- Wave32 16x16x128 f8f6f4 sub-byte (fp6/bf6/fp4) WMMA dataflow -----------
        // The f8f6f4 WMMA reaches sub-byte operands (fp6/bf6 6-bit, fp4 4-bit) that
        // cannot be Value scalar element types, so each per-lane operand is the raw
        // 512-bit packed register (v16i32 == int[16]) the builtin consumes. In FPSan
        // mode its bits ARE the per-slot payloads: slot s occupies the contiguous
        // Width-bit field at bit Width*s (little-endian across the 16 words), and the
        // FPSan widen of a Width-bit code is the signed resize (subbyte_widen), exactly
        // as the fp4/fp6 cvt-unpack path. The cross-lane element->(lane, slot) mapping
        // is the SAME Wmma16x16x128Layout validated for the 8-bit K=128 WMMA: the
        // grounding test confirms bit-exact builtin equality for every same-class
        // (8x8, sub x sub incl. fp6 x fp4) staging. Mixing an 8-bit with a sub-byte
        // operand uses a DIFFERENT physical k order on hardware and is NOT handled here
        // (see the DISABLED mixed cases in the grounding test).

        // Gather the Width-bit field at slot s from `srclane`'s packed 512-bit register
        // (16 i32, this lane's copy) and widen it to an f32 payload. The field may
        // straddle two 32-bit words for fp6 (6 does not divide 32).
        template <int Width, Semantics S, Conversions C>
        FPSAN_DEVICE Value<float, S, C>
                     wmma_sub_gather_widen(const int (&pw)[16], int s, int srclane)
        {
            const int p = Width * s, wi = p >> 5, off = p & 31;
            unsigned  w0 = static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(srclane * 4, pw[wi]));
            unsigned  field = w0 >> off;
            if(off > 32 - Width)
            {
                unsigned w1
                    = static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(srclane * 4, pw[wi + 1]));
                field |= w1 << (32 - off);
            }
            field &= (1u << Width) - 1u;
            return subbyte_widen<Width, S, C>(field);
        }

        // Wave-cooperative software MMA for the gfx1250 16x16x128 f8f6f4 sub-byte
        // shapes (K=128, fp6/bf6/fp4 A/B operands of possibly different sub-byte
        // widths, v8f C/D accumulator). FPSan-only (Float mode calls the builtin).
        template <int WA, int WB, Semantics S, Conversions C>
        FPSAN_DEVICE Value<v8f_native, S, C> wmma_16x16x128_sub_dataflow(const int (&aw)[16],
                                                                         const int (&bw)[16],
                                                                         Value<v8f_native, S, C> c)
        {
            using Acc                    = Value<float, S, C>;
            const int               lane = wave_lane();
            const int               n    = Wmma16x16x128Layout::cd_n(lane);
            Value<v8f_native, S, C> d{};
            for(int e = 0; e < 8; ++e)
            {
                const int m   = Wmma16x16x128Layout::cd_m(lane, e);
                Acc       acc = c.get(e);
                for(int k = 0; k < 128; ++k)
                {
                    const int slot = Wmma16x16x128Layout::ab_index(k);
                    auto      av   = wmma_sub_gather_widen<WA, S, C>(
                        aw, slot, Wmma16x16x128Layout::ab_lane(m, k));
                    auto bv = wmma_sub_gather_widen<WB, S, C>(
                        bw, slot, Wmma16x16x128Layout::ab_lane(n, k));
                    acc = acc + av * bv;
                }
                d.set(e, acc);
            }
            return d;
        }

        // ---- Wave32 16x16x128 f8f6f4 MIXED (8-bit x sub-byte) WMMA dataflow ---------
        // When one operand is 8-bit (fp8/bf8) and the other sub-byte (fp6/bf6/fp4),
        // the two operands occupy DIFFERENT physical k orders in the per-lane
        // fragment, so the unified same-class staging no longer cancels. The 8-bit
        // operand keeps the validated Wmma16x16x128Layout (ab_lane/ab_index); the
        // sub-byte operand follows the order below, reverse-engineered from one-hot
        // probes against the public builtin (sub element for logical k lands at the
        // lane/slot given here; verified bit-exact 128/128 for both operand roles):
        //   lane = (row|col) + 16*((k>>5)&1)   (uses k bit 5, vs the 8-bit bit 2)
        //   slot = 32*((k>>6)&1) + 16*((k>>2)&1) + 8*((k>>4)&1)
        //          + 4*((k>>3)&1) + 2*((k>>1)&1) + (k&1)
        // FPSan-only (Float mode calls the builtin, which is layout-agnostic).
        FPSAN_HOST_DEVICE constexpr int wmma_mix_sub_lane(int rc, int k)
        {
            return rc + 16 * ((k >> 5) & 1);
        }
        FPSAN_HOST_DEVICE constexpr int wmma_mix_sub_slot(int k)
        {
            return 32 * ((k >> 6) & 1) + 16 * ((k >> 2) & 1) + 8 * ((k >> 4) & 1)
                   + 4 * ((k >> 3) & 1) + 2 * ((k >> 1) & 1) + (k & 1);
        }

        template <int WA, int WB, Semantics S, Conversions C>
        FPSAN_DEVICE Value<v8f_native, S, C> wmma_16x16x128_mixed_dataflow(
            const int (&aw)[16], const int (&bw)[16], Value<v8f_native, S, C> c)
        {
            static_assert((WA == 8) != (WB == 8),
                          "mixed dataflow: exactly one operand must be 8-bit, the other sub-byte");
            using Acc                    = Value<float, S, C>;
            const int               lane = wave_lane();
            const int               n    = Wmma16x16x128Layout::cd_n(lane);
            Value<v8f_native, S, C> d{};
            for(int e = 0; e < 8; ++e)
            {
                const int m   = Wmma16x16x128Layout::cd_m(lane, e);
                Acc       acc = c.get(e);
                for(int k = 0; k < 128; ++k)
                {
                    const int aslot
                        = (WA == 8) ? Wmma16x16x128Layout::ab_index(k) : wmma_mix_sub_slot(k);
                    const int alane
                        = (WA == 8) ? Wmma16x16x128Layout::ab_lane(m, k) : wmma_mix_sub_lane(m, k);
                    const int bslot
                        = (WB == 8) ? Wmma16x16x128Layout::ab_index(k) : wmma_mix_sub_slot(k);
                    const int blane
                        = (WB == 8) ? Wmma16x16x128Layout::ab_lane(n, k) : wmma_mix_sub_lane(n, k);
                    auto av = wmma_sub_gather_widen<WA, S, C>(aw, aslot, alane);
                    auto bv = wmma_sub_gather_widen<WB, S, C>(bw, bslot, blane);
                    acc     = acc + av * bv;
                }
                d.set(e, acc);
            }
            return d;
        }

        // Sub-byte (fp6/bf6/fp4) scale-byte index for logical K element `k`: which
        // of the per-lane i32 scale operand's 4 E8M0 bytes scales this element.
        // Silicon-grounded (wmma_scale_f8f6f4_gfx1250_test probe, identical for
        // fp6/bf6/fp4): byte = 2*(k>>6) + ((k>>2)&1) -- the high K half picks the
        // byte pair and the fragment lane-half ((k>>2)&1) picks within the pair.
        // (This differs from the 8-bit mapping byte = k>>5.)
        FPSAN_HOST_DEVICE constexpr int wmma_sub_scale_byte(int k)
        {
            return 2 * (k >> 6) + ((k >> 2) & 1);
        }

        // scale16 (i64, 8 E8M0 bytes / finer granularity) sub-byte scale-byte index.
        // Silicon-grounded (wmma_scale_f8f6f4_gfx1250_test probe, identical for
        // fp6/bf6/fp4): byte = 4*(k>>6) + 2*((k>>2)&1) + ((k>>5)&1) -- adds the
        // (k>>5) sub-block bit to the i32 map. (8-bit scale16 differs:
        // byte = 2*(k>>5) + ((k>>2)&1).)
        FPSAN_HOST_DEVICE constexpr int wmma_sub_scale16_byte(int k)
        {
            return 4 * (k >> 6) + 2 * ((k >> 2) & 1) + ((k >> 5) & 1);
        }

        // MIXED 8-bit x sub-byte block-scale byte maps. Silicon-grounded
        // (probe_mixscale one-hot sweep, see wmma_16x16x128_f8f6f4 mixed tests):
        // unlike the same-class sub maps above, in a mix BOTH operands follow the
        // 8-bit scale convention -- i32 byte = k>>5 (4 blocks of 32), i64/scale16
        // byte = 2*(k>>5) + ((k>>2)&1) (8 sub-blocks). Verified identical for the
        // 8-bit and the sub-byte operand role.
        FPSAN_HOST_DEVICE constexpr int wmma_mix_scale_byte(int k)
        {
            return k >> 5;
        }
        FPSAN_HOST_DEVICE constexpr int wmma_mix_scale16_byte(int k)
        {
            return 2 * (k >> 5) + ((k >> 2) & 1);
        }

        // Decode one block-scale byte per the WMMA scale-format immediate. The
        // gfx1250 scaled WMMA carries a per-operand scale FORMAT (the 2nd of each
        // sel/fmt immediate pair); the accepted FP scale formats are
        //   0 = E8M0FNU (exponent-only micro-scale, 2^(b-127)), and
        //   2 = E4M3FN  (full fp8 micro-scale, mantissa included).
        // Silicon-grounded: with fmt=2 a scale byte of E4M3(2.0) doubles the
        // block product (vs the ~2^-63 an E8M0 read would give). E5M3 is reserved by
        // the HW enum but unencoded by the toolchain, so it is intentionally not
        // exposed. E4M3 scales are only legal in matched configs
        // (both-E4M3 f4xf4, or the f4 side of an f4 x f{8,6} mix); this decoder is
        // the per-byte primitive, the wrappers enforce the legal combinations.
        template <int ScaleFmt>
        FPSAN_HOST_DEVICE inline float wmma_scale_byte_to_float(unsigned byte)
        {
            static_assert(ScaleFmt == 0 || ScaleFmt == 2,
                          "WMMA scale format: 0 = E8M0FNU, 2 = E4M3FN");
            if constexpr(ScaleFmt == 2)
                return narrow_to_f32(byte & 0xFFu, kFp8E4M3);
            else
                return e8m0_to_float(byte);
        }

        // ---- Wave32 16x16x128 f8f6f4 sub-byte BLOCK-SCALED WMMA dataflow ------------
        // The block-scaled f8f6f4 variant: each K element carries a per-(row/col,
        // block) E8M0 scale and its product is multiplied by 2^(eA-127)*2^(eB-127).
        // Scale-operand layout is the grounded wave32 model (validated bit-exact vs
        // the builtin in wmma_scale_f8f6f4_gfx1250_test): the scale word for row m /
        // col n lives at lane m / lane n (fetched with ds_bpermute), and within that
        // word the byte is wmma_sub_scale_byte(k) (scaleType/scaleFmt immediates
        // pinned 0 == E8M0). FPSan-only (Float mode calls the builtin).
        template <int WA, int WB, Semantics S, Conversions C, int AScaleFmt = 0, int BScaleFmt = 0>
        FPSAN_DEVICE Value<v8f_native, S, C> wmma_16x16x128_sub_scaled_dataflow(
            const int (&aw)[16], const int (&bw)[16], Value<v8f_native, S, C> c, int sa, int sb)
        {
            using Acc                    = Value<float, S, C>;
            const int               lane = wave_lane();
            const int               n    = Wmma16x16x128Layout::cd_n(lane);
            Value<v8f_native, S, C> d{};
            for(int e = 0; e < 8; ++e)
            {
                const int      m   = Wmma16x16x128Layout::cd_m(lane, e);
                const unsigned wa  = static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(m * 4, sa));
                const unsigned wb  = static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(n * 4, sb));
                Acc            acc = c.get(e);
                for(int k = 0; k < 128; ++k)
                {
                    const int slot = Wmma16x16x128Layout::ab_index(k);
                    auto      av   = wmma_sub_gather_widen<WA, S, C>(
                        aw, slot, Wmma16x16x128Layout::ab_lane(m, k));
                    auto bv = wmma_sub_gather_widen<WB, S, C>(
                        bw, slot, Wmma16x16x128Layout::ab_lane(n, k));
                    const int byte = wmma_sub_scale_byte(k);
                    const Acc fa(wmma_scale_byte_to_float<AScaleFmt>((wa >> (8 * byte)) & 0xFFu));
                    const Acc fb(wmma_scale_byte_to_float<BScaleFmt>((wb >> (8 * byte)) & 0xFFu));
                    acc = acc + av * bv * fa * fb;
                }
                d.set(e, acc);
            }
            return d;
        }

        // ---- Wave32 16x16x128 f8f6f4 sub-byte scale16 (i64) WMMA dataflow ----------
        // As wmma_16x16x128_sub_scaled_dataflow, but the per-operand scale operand is
        // a 64-bit word (8 E8M0 bytes, finer granularity); the controlling byte is
        // wmma_sub_scale16_byte(k). The 64-bit scale word for row m / col n is
        // gathered from lane m / lane n with a pair of 32-bit ds_bpermutes.
        template <int WA, int WB, Semantics S, Conversions C, int AScaleFmt = 0, int BScaleFmt = 0>
        FPSAN_DEVICE Value<v8f_native, S, C>
                     wmma_16x16x128_sub_scaled16_dataflow(const int (&aw)[16],
                                                          const int (&bw)[16],
                                                          Value<v8f_native, S, C> c,
                                                          long long               sa,
                                                          long long               sb)
        {
            using Acc                     = Value<float, S, C>;
            const int               lane  = wave_lane();
            const int               n     = Wmma16x16x128Layout::cd_n(lane);
            const int               sa_lo = static_cast<int>(sa & 0xffffffff);
            const int               sa_hi = static_cast<int>((sa >> 32) & 0xffffffff);
            const int               sb_lo = static_cast<int>(sb & 0xffffffff);
            const int               sb_hi = static_cast<int>((sb >> 32) & 0xffffffff);
            Value<v8f_native, S, C> d{};
            for(int e = 0; e < 8; ++e)
            {
                const int                m = Wmma16x16x128Layout::cd_m(lane, e);
                const unsigned long long wa
                    = (static_cast<unsigned long long>(
                           static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(m * 4, sa_lo)))
                       | (static_cast<unsigned long long>(
                              static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(m * 4, sa_hi)))
                          << 32));
                const unsigned long long wb
                    = (static_cast<unsigned long long>(
                           static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(n * 4, sb_lo)))
                       | (static_cast<unsigned long long>(
                              static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(n * 4, sb_hi)))
                          << 32));
                Acc acc = c.get(e);
                for(int k = 0; k < 128; ++k)
                {
                    const int slot = Wmma16x16x128Layout::ab_index(k);
                    auto      av   = wmma_sub_gather_widen<WA, S, C>(
                        aw, slot, Wmma16x16x128Layout::ab_lane(m, k));
                    auto bv = wmma_sub_gather_widen<WB, S, C>(
                        bw, slot, Wmma16x16x128Layout::ab_lane(n, k));
                    const int byte = wmma_sub_scale16_byte(k);
                    const Acc fa(wmma_scale_byte_to_float<AScaleFmt>(
                        static_cast<unsigned>((wa >> (8 * byte)) & 0xFFull)));
                    const Acc fb(wmma_scale_byte_to_float<BScaleFmt>(
                        static_cast<unsigned>((wb >> (8 * byte)) & 0xFFull)));
                    acc = acc + av * bv * fa * fb;
                }
                d.set(e, acc);
            }
            return d;
        }

        // ---- Wave32 16x16x128 f8f6f4 MIXED 8-bit x sub-byte BLOCK-SCALED dataflow ---
        // Combines the mixed operand staging (8-bit on Wmma16x16x128Layout, sub-byte
        // on wmma_mix_sub_lane/slot) with per-(row/col, block) E8M0/E4M3 scaling. In
        // a mix BOTH operands use the 8-bit scale-byte map wmma_mix_scale_byte(k)
        // (i32) -- distinct from the same-class sub map. The scale word for row m /
        // col n is the i32 supplied at lane m / lane n (ds_bpermute). FPSan-only.
        template <int WA, int WB, Semantics S, Conversions C, int AScaleFmt = 0, int BScaleFmt = 0>
        FPSAN_DEVICE Value<v8f_native, S, C> wmma_16x16x128_mixed_scaled_dataflow(
            const int (&aw)[16], const int (&bw)[16], Value<v8f_native, S, C> c, int sa, int sb)
        {
            static_assert((WA == 8) != (WB == 8),
                          "mixed scaled dataflow: exactly one operand must be 8-bit");
            using Acc                    = Value<float, S, C>;
            const int               lane = wave_lane();
            const int               n    = Wmma16x16x128Layout::cd_n(lane);
            Value<v8f_native, S, C> d{};
            for(int e = 0; e < 8; ++e)
            {
                const int      m   = Wmma16x16x128Layout::cd_m(lane, e);
                const unsigned wa  = static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(m * 4, sa));
                const unsigned wb  = static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(n * 4, sb));
                Acc            acc = c.get(e);
                for(int k = 0; k < 128; ++k)
                {
                    const int aslot
                        = (WA == 8) ? Wmma16x16x128Layout::ab_index(k) : wmma_mix_sub_slot(k);
                    const int alane
                        = (WA == 8) ? Wmma16x16x128Layout::ab_lane(m, k) : wmma_mix_sub_lane(m, k);
                    const int bslot
                        = (WB == 8) ? Wmma16x16x128Layout::ab_index(k) : wmma_mix_sub_slot(k);
                    const int blane
                        = (WB == 8) ? Wmma16x16x128Layout::ab_lane(n, k) : wmma_mix_sub_lane(n, k);
                    auto      av   = wmma_sub_gather_widen<WA, S, C>(aw, aslot, alane);
                    auto      bv   = wmma_sub_gather_widen<WB, S, C>(bw, bslot, blane);
                    const int byte = wmma_mix_scale_byte(k);
                    const Acc fa(wmma_scale_byte_to_float<AScaleFmt>((wa >> (8 * byte)) & 0xFFu));
                    const Acc fb(wmma_scale_byte_to_float<BScaleFmt>((wb >> (8 * byte)) & 0xFFu));
                    acc = acc + av * bv * fa * fb;
                }
                d.set(e, acc);
            }
            return d;
        }

        // scale16 (i64, 8 bytes) variant of the mixed scaled dataflow; controlling
        // byte = wmma_mix_scale16_byte(k). The 64-bit scale word for row m / col n is
        // gathered from lane m / lane n with a pair of 32-bit ds_bpermutes.
        template <int WA, int WB, Semantics S, Conversions C, int AScaleFmt = 0, int BScaleFmt = 0>
        FPSAN_DEVICE Value<v8f_native, S, C>
                     wmma_16x16x128_mixed_scaled16_dataflow(const int (&aw)[16],
                                                            const int (&bw)[16],
                                                            Value<v8f_native, S, C> c,
                                                            long long               sa,
                                                            long long               sb)
        {
            static_assert((WA == 8) != (WB == 8),
                          "mixed scaled16 dataflow: exactly one operand must be 8-bit");
            using Acc                     = Value<float, S, C>;
            const int               lane  = wave_lane();
            const int               n     = Wmma16x16x128Layout::cd_n(lane);
            const int               sa_lo = static_cast<int>(sa & 0xffffffff);
            const int               sa_hi = static_cast<int>((sa >> 32) & 0xffffffff);
            const int               sb_lo = static_cast<int>(sb & 0xffffffff);
            const int               sb_hi = static_cast<int>((sb >> 32) & 0xffffffff);
            Value<v8f_native, S, C> d{};
            for(int e = 0; e < 8; ++e)
            {
                const int                m = Wmma16x16x128Layout::cd_m(lane, e);
                const unsigned long long wa
                    = (static_cast<unsigned long long>(
                           static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(m * 4, sa_lo)))
                       | (static_cast<unsigned long long>(
                              static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(m * 4, sa_hi)))
                          << 32));
                const unsigned long long wb
                    = (static_cast<unsigned long long>(
                           static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(n * 4, sb_lo)))
                       | (static_cast<unsigned long long>(
                              static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(n * 4, sb_hi)))
                          << 32));
                Acc acc = c.get(e);
                for(int k = 0; k < 128; ++k)
                {
                    const int aslot
                        = (WA == 8) ? Wmma16x16x128Layout::ab_index(k) : wmma_mix_sub_slot(k);
                    const int alane
                        = (WA == 8) ? Wmma16x16x128Layout::ab_lane(m, k) : wmma_mix_sub_lane(m, k);
                    const int bslot
                        = (WB == 8) ? Wmma16x16x128Layout::ab_index(k) : wmma_mix_sub_slot(k);
                    const int blane
                        = (WB == 8) ? Wmma16x16x128Layout::ab_lane(n, k) : wmma_mix_sub_lane(n, k);
                    auto      av   = wmma_sub_gather_widen<WA, S, C>(aw, aslot, alane);
                    auto      bv   = wmma_sub_gather_widen<WB, S, C>(bw, bslot, blane);
                    const int byte = wmma_mix_scale16_byte(k);
                    const Acc fa(wmma_scale_byte_to_float<AScaleFmt>(
                        static_cast<unsigned>((wa >> (8 * byte)) & 0xFFull)));
                    const Acc fb(wmma_scale_byte_to_float<BScaleFmt>(
                        static_cast<unsigned>((wb >> (8 * byte)) & 0xFFull)));
                    acc = acc + av * bv * fa * fb;
                }
                d.set(e, acc);
            }
            return d;
        }

        // ---- Wave32 32x16x128 f4 (E2M1) WMMA layout + dataflow ----------------------
        // Silicon-grounded fragment ABI (wmma_32x16x128_f4_gfx1250_test one-hot +
        // binary-k probes). A is 32x128 fp4 (v16i/lane), B is 128x16 fp4 (v8i/lane),
        // C/D is 32x16 f32 (v16f/lane).
        //   A[m][k]: g=m>>3; L=m-8*((g+1)>>1); lane=L+16*((k>>2)&1);
        //            slot=64*((m>>3)&1)+(k&3)+4*(k>>3)
        //   B[k][n]: lane=n+16*((k>>2)&1); slot reuses the 16x16x128 ab_index
        //   D[m][n]: lane=n+16*(m>>4); reg=m&15
        struct Wmma32x16x128F4Layout
        {
            FPSAN_DEVICE static int a_lane(int m, int k)
            {
                const int g = m >> 3;
                const int L = m - 8 * ((g + 1) >> 1);
                return L + 16 * ((k >> 2) & 1);
            }
            FPSAN_DEVICE static int a_slot(int m, int k)
            {
                return 64 * ((m >> 3) & 1) + (k & 3) + 4 * (k >> 3);
            }
            FPSAN_DEVICE static int b_lane(int n, int k)
            {
                return n + 16 * ((k >> 2) & 1);
            }
            FPSAN_DEVICE static int b_slot(int k)
            {
                const int reg = ((k >> 1) & 1) + 2 * ((k >> 3) & 1) + 4 * ((k >> 4) & 1)
                                + 8 * ((k >> 5) & 1) + 16 * ((k >> 6) & 1);
                return 2 * reg + (k & 1);
            }
            FPSAN_DEVICE static int cd_m(int lane, int reg)
            {
                return reg + 16 * (lane >> 4);
            }
            FPSAN_DEVICE static int cd_n(int lane)
            {
                return lane & 15;
            }
            // Per-operand block-scale source lanes (silicon-grounded). The A-scale
            // word for row m lives in lane scale_lane_a(m) (octets 1 and 2 swap
            // lane ranges vs the row index); the B-scale word for col n lives in
            // lane n. Controlling byte = detail::wmma_sub_scale_byte(k) (i32) /
            // wmma_sub_scale16_byte(k) (i64), the same maps as 16x16x128.
            FPSAN_DEVICE static int scale_lane_a(int m)
            {
                const int g = m >> 3;
                return (g == 1) ? (m + 8) : (g == 2) ? (m - 8) : m;
            }
        };

        // FPSan software MMA for gfx1250 wmma_f32_32x16x128_f4 (Float mode calls the
        // builtin). fp4 nibbles never span a dword (width 4), so the B operand's 8
        // dwords fit the [16] gather array zero-padded.
        template <Semantics S, Conversions C>
        FPSAN_DEVICE Value<v16f_native, S, C> wmma_32x16x128_f4_dataflow(const int (&aw)[16],
                                                                         const int (&bw)[16],
                                                                         Value<v16f_native, S, C> c)
        {
            using Acc                     = Value<float, S, C>;
            const int                lane = wave_lane();
            const int                n    = Wmma32x16x128F4Layout::cd_n(lane);
            Value<v16f_native, S, C> d{};
            for(int e = 0; e < 16; ++e)
            {
                const int m   = Wmma32x16x128F4Layout::cd_m(lane, e);
                Acc       acc = c.get(e);
                for(int k = 0; k < 128; ++k)
                {
                    auto av = wmma_sub_gather_widen<4, S, C>(aw,
                                                             Wmma32x16x128F4Layout::a_slot(m, k),
                                                             Wmma32x16x128F4Layout::a_lane(m, k));
                    auto bv = wmma_sub_gather_widen<4, S, C>(
                        bw, Wmma32x16x128F4Layout::b_slot(k), Wmma32x16x128F4Layout::b_lane(n, k));
                    acc = acc + av * bv;
                }
                d.set(e, acc);
            }
            return d;
        }

        // ---- 32x16x128 f4 BLOCK-SCALED (i32 / i64) dataflow -------------------------
        // A-scale word for row m gathered from lane scale_lane_a(m); B-scale word for
        // col n from lane n. ScaleI is the per-lane operand width (4 -> i32, 8 -> i64
        // E8M0 bytes); ByteFn maps k -> controlling byte.
        template <Semantics S, Conversions C, int AScaleFmt = 0, int BScaleFmt = 0>
        FPSAN_DEVICE Value<v16f_native, S, C> wmma_32x16x128_f4_scaled_dataflow(
            const int (&aw)[16], const int (&bw)[16], Value<v16f_native, S, C> c, int sa, int sb)
        {
            using Acc           = Value<float, S, C>;
            const int      lane = wave_lane();
            const int      n    = Wmma32x16x128F4Layout::cd_n(lane);
            const unsigned wb   = static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(n * 4, sb));
            Value<v16f_native, S, C> d{};
            for(int e = 0; e < 16; ++e)
            {
                const int      m  = Wmma32x16x128F4Layout::cd_m(lane, e);
                const unsigned wa = static_cast<unsigned>(
                    __builtin_amdgcn_ds_bpermute(Wmma32x16x128F4Layout::scale_lane_a(m) * 4, sa));
                Acc acc = c.get(e);
                for(int k = 0; k < 128; ++k)
                {
                    auto av = wmma_sub_gather_widen<4, S, C>(aw,
                                                             Wmma32x16x128F4Layout::a_slot(m, k),
                                                             Wmma32x16x128F4Layout::a_lane(m, k));
                    auto bv = wmma_sub_gather_widen<4, S, C>(
                        bw, Wmma32x16x128F4Layout::b_slot(k), Wmma32x16x128F4Layout::b_lane(n, k));
                    const int byte = wmma_sub_scale_byte(k);
                    const Acc fa(wmma_scale_byte_to_float<AScaleFmt>((wa >> (8 * byte)) & 0xFFu));
                    const Acc fb(wmma_scale_byte_to_float<BScaleFmt>((wb >> (8 * byte)) & 0xFFu));
                    acc = acc + av * bv * fa * fb;
                }
                d.set(e, acc);
            }
            return d;
        }

        template <Semantics S, Conversions C, int AScaleFmt = 0, int BScaleFmt = 0>
        FPSAN_DEVICE Value<v16f_native, S, C>
                     wmma_32x16x128_f4_scaled16_dataflow(const int (&aw)[16],
                                                         const int (&bw)[16],
                                                         Value<v16f_native, S, C> c,
                                                         long long                sa,
                                                         long long                sb)
        {
            using Acc                      = Value<float, S, C>;
            const int                lane  = wave_lane();
            const int                n     = Wmma32x16x128F4Layout::cd_n(lane);
            const int                sa_lo = static_cast<int>(sa & 0xffffffff);
            const int                sa_hi = static_cast<int>((sa >> 32) & 0xffffffff);
            const int                sb_lo = static_cast<int>(sb & 0xffffffff);
            const int                sb_hi = static_cast<int>((sb >> 32) & 0xffffffff);
            const unsigned long long wb
                = (static_cast<unsigned long long>(
                       static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(n * 4, sb_lo)))
                   | (static_cast<unsigned long long>(
                          static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(n * 4, sb_hi)))
                      << 32));
            Value<v16f_native, S, C> d{};
            for(int e = 0; e < 16; ++e)
            {
                const int                m  = Wmma32x16x128F4Layout::cd_m(lane, e);
                const int                la = Wmma32x16x128F4Layout::scale_lane_a(m) * 4;
                const unsigned long long wa
                    = (static_cast<unsigned long long>(
                           static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(la, sa_lo)))
                       | (static_cast<unsigned long long>(
                              static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(la, sa_hi)))
                          << 32));
                Acc acc = c.get(e);
                for(int k = 0; k < 128; ++k)
                {
                    auto av = wmma_sub_gather_widen<4, S, C>(aw,
                                                             Wmma32x16x128F4Layout::a_slot(m, k),
                                                             Wmma32x16x128F4Layout::a_lane(m, k));
                    auto bv = wmma_sub_gather_widen<4, S, C>(
                        bw, Wmma32x16x128F4Layout::b_slot(k), Wmma32x16x128F4Layout::b_lane(n, k));
                    const int byte = wmma_sub_scale16_byte(k);
                    const Acc fa(wmma_scale_byte_to_float<AScaleFmt>(
                        static_cast<unsigned>((wa >> (8 * byte)) & 0xFFull)));
                    const Acc fb(wmma_scale_byte_to_float<BScaleFmt>(
                        static_cast<unsigned>((wb >> (8 * byte)) & 0xFFull)));
                    acc = acc + av * bv * fa * fb;
                }
                d.set(e, acc);
            }
            return d;
        }

    } // namespace detail

// =============================================================================
// RDNA3 (gfx11) wave32 WMMA wrappers. These are the base Clang builtins without
// the gfx12 suffix. A/B operands are v16 replicated fragments; f32 accumulators
// use v8f, while f16/bf16 accumulators use v16 with OPSEL selecting the active
// half-register. RDNA3 does not expose the gfx12 fp8/bf8 WMMA family.
// =============================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX11__)

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<v8f_native, S, C> amdgcn_wmma_f32_16x16x16_f16_w32(
        Value<v16h_wmma_native, S, C> a, Value<v16h_wmma_native, S, C> b, Value<v8f_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v8f_native d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                a.to_float(), b.to_float(), c.to_float());
            return Value<v8f_native, S, C>(d);
        }
        else
        {
            return detail::wmma_gfx11_f32_16x16x16_dataflow(a, b, c);
        }
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<v8f_native, S, C>
                 amdgcn_wmma_f32_16x16x16_bf16_w32(Value<v16bf_wmma_native, S, C> a,
                                                   Value<v16bf_wmma_native, S, C> b,
                                                   Value<v8f_native, S, C>        c)
    {
        if constexpr(S == Semantics::Native)
        {
            const v16i16_wmma_native ai = __builtin_bit_cast(v16i16_wmma_native, a.to_float());
            const v16i16_wmma_native bi = __builtin_bit_cast(v16i16_wmma_native, b.to_float());
            v8f_native d = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ai, bi, c.to_float());
            return Value<v8f_native, S, C>(d);
        }
        else
        {
            return detail::wmma_gfx11_f32_16x16x16_dataflow(a, b, c);
        }
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f16_16x16x16_f16_w32)
    template <bool        Opsel = false,
              Semantics   S     = Semantics::Native,
              Conversions C     = Conversions::Explicit>
    FPSAN_DEVICE Value<v16h_wmma_native, S, C>
                 amdgcn_wmma_f16_16x16x16_f16_w32(Value<v16h_wmma_native, S, C> a,
                                                  Value<v16h_wmma_native, S, C> b,
                                                  Value<v16h_wmma_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v16h_wmma_native d = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
                a.to_float(), b.to_float(), c.to_float(), Opsel);
            return Value<v16h_wmma_native, S, C>(d);
        }
        else
        {
            return detail::wmma_gfx11_half_16x16x16_dataflow<Opsel>(a, b, c);
        }
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) \
    || __has_builtin(__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32)
    template <bool        Opsel = false,
              Semantics   S     = Semantics::Native,
              Conversions C     = Conversions::Explicit>
    FPSAN_DEVICE Value<v16h_wmma_native, S, C>
                 amdgcn_wmma_f16_16x16x16_f16_tied_w32(Value<v16h_wmma_native, S, C> a,
                                                       Value<v16h_wmma_native, S, C> b,
                                                       Value<v16h_wmma_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v16h_wmma_native d = __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32(
                a.to_float(), b.to_float(), c.to_float(), Opsel);
            return Value<v16h_wmma_native, S, C>(d);
        }
        else
        {
            return detail::wmma_gfx11_half_16x16x16_dataflow<Opsel, true>(a, b, c);
        }
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32)
    template <bool        Opsel = false,
              Semantics   S     = Semantics::Native,
              Conversions C     = Conversions::Explicit>
    FPSAN_DEVICE Value<v16bf_wmma_native, S, C>
                 amdgcn_wmma_bf16_16x16x16_bf16_w32(Value<v16bf_wmma_native, S, C> a,
                                                    Value<v16bf_wmma_native, S, C> b,
                                                    Value<v16bf_wmma_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            const v16i16_wmma_native ai = __builtin_bit_cast(v16i16_wmma_native, a.to_float());
            const v16i16_wmma_native bi = __builtin_bit_cast(v16i16_wmma_native, b.to_float());
            const v16i16_wmma_native ci = __builtin_bit_cast(v16i16_wmma_native, c.to_float());
            const v16i16_wmma_native di
                = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(ai, bi, ci, Opsel);
            return Value<v16bf_wmma_native, S, C>(__builtin_bit_cast(v16bf_wmma_native, di));
        }
        else
        {
            return detail::wmma_gfx11_half_16x16x16_dataflow<Opsel>(a, b, c);
        }
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) \
    || __has_builtin(__builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w32)
    template <bool        Opsel = false,
              Semantics   S     = Semantics::Native,
              Conversions C     = Conversions::Explicit>
    FPSAN_DEVICE Value<v16bf_wmma_native, S, C>
                 amdgcn_wmma_bf16_16x16x16_bf16_tied_w32(Value<v16bf_wmma_native, S, C> a,
                                                         Value<v16bf_wmma_native, S, C> b,
                                                         Value<v16bf_wmma_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            const v16i16_wmma_native ai = __builtin_bit_cast(v16i16_wmma_native, a.to_float());
            const v16i16_wmma_native bi = __builtin_bit_cast(v16i16_wmma_native, b.to_float());
            const v16i16_wmma_native ci = __builtin_bit_cast(v16i16_wmma_native, c.to_float());
            const v16i16_wmma_native di
                = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w32(ai, bi, ci, Opsel);
            return Value<v16bf_wmma_native, S, C>(__builtin_bit_cast(v16bf_wmma_native, di));
        }
        else
        {
            return detail::wmma_gfx11_half_16x16x16_dataflow<Opsel, true>(a, b, c);
        }
    }
#endif

// gfx11 wave64 variants. The builtins share A/B with wave32 but use the smaller
// C/D ABI implied by four rows per accumulator register.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w64)
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<v4f_native, S, C> amdgcn_wmma_f32_16x16x16_f16_w64(
        Value<v16h_wmma_native, S, C> a, Value<v16h_wmma_native, S, C> b, Value<v4f_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v4f_native d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w64(
                a.to_float(), b.to_float(), c.to_float());
            return Value<v4f_native, S, C>(d);
        }
        else
        {
            return detail::wmma_gfx11_f32_16x16x16_dataflow(a, b, c);
        }
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w64)
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<v4f_native, S, C>
                 amdgcn_wmma_f32_16x16x16_bf16_w64(Value<v16bf_wmma_native, S, C> a,
                                                   Value<v16bf_wmma_native, S, C> b,
                                                   Value<v4f_native, S, C>        c)
    {
        if constexpr(S == Semantics::Native)
        {
            const v16i16_wmma_native ai = __builtin_bit_cast(v16i16_wmma_native, a.to_float());
            const v16i16_wmma_native bi = __builtin_bit_cast(v16i16_wmma_native, b.to_float());
            v4f_native d = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64(ai, bi, c.to_float());
            return Value<v4f_native, S, C>(d);
        }
        else
        {
            return detail::wmma_gfx11_f32_16x16x16_dataflow(a, b, c);
        }
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f16_16x16x16_f16_w64)
    template <bool        Opsel = false,
              Semantics   S     = Semantics::Native,
              Conversions C     = Conversions::Explicit>
    FPSAN_DEVICE Value<v8h_native, S, C> amdgcn_wmma_f16_16x16x16_f16_w64(
        Value<v16h_wmma_native, S, C> a, Value<v16h_wmma_native, S, C> b, Value<v8h_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v8h_native d = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                a.to_float(), b.to_float(), c.to_float(), Opsel);
            return Value<v8h_native, S, C>(d);
        }
        else
        {
            return detail::wmma_gfx11_half_16x16x16_dataflow<Opsel>(a, b, c);
        }
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) \
    || __has_builtin(__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w64)
    template <bool        Opsel = false,
              Semantics   S     = Semantics::Native,
              Conversions C     = Conversions::Explicit>
    FPSAN_DEVICE Value<v8h_native, S, C> amdgcn_wmma_f16_16x16x16_f16_tied_w64(
        Value<v16h_wmma_native, S, C> a, Value<v16h_wmma_native, S, C> b, Value<v8h_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v8h_native d = __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w64(
                a.to_float(), b.to_float(), c.to_float(), Opsel);
            return Value<v8h_native, S, C>(d);
        }
        else
        {
            return detail::wmma_gfx11_half_16x16x16_dataflow<Opsel, true>(a, b, c);
        }
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64)
    template <bool        Opsel = false,
              Semantics   S     = Semantics::Native,
              Conversions C     = Conversions::Explicit>
    FPSAN_DEVICE Value<v8bf_native, S, C>
                 amdgcn_wmma_bf16_16x16x16_bf16_w64(Value<v16bf_wmma_native, S, C> a,
                                                    Value<v16bf_wmma_native, S, C> b,
                                                    Value<v8bf_native, S, C>       c)
    {
        if constexpr(S == Semantics::Native)
        {
            const v16i16_wmma_native ai = __builtin_bit_cast(v16i16_wmma_native, a.to_float());
            const v16i16_wmma_native bi = __builtin_bit_cast(v16i16_wmma_native, b.to_float());
            const v8i16_wmma_native  ci = __builtin_bit_cast(v8i16_wmma_native, c.to_float());
            const v8i16_wmma_native  di
                = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64(ai, bi, ci, Opsel);
            return Value<v8bf_native, S, C>(__builtin_bit_cast(v8bf_native, di));
        }
        else
        {
            return detail::wmma_gfx11_half_16x16x16_dataflow<Opsel>(a, b, c);
        }
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) \
    || __has_builtin(__builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w64)
    template <bool        Opsel = false,
              Semantics   S     = Semantics::Native,
              Conversions C     = Conversions::Explicit>
    FPSAN_DEVICE Value<v8bf_native, S, C>
                 amdgcn_wmma_bf16_16x16x16_bf16_tied_w64(Value<v16bf_wmma_native, S, C> a,
                                                         Value<v16bf_wmma_native, S, C> b,
                                                         Value<v8bf_native, S, C>       c)
    {
        if constexpr(S == Semantics::Native)
        {
            const v16i16_wmma_native ai = __builtin_bit_cast(v16i16_wmma_native, a.to_float());
            const v16i16_wmma_native bi = __builtin_bit_cast(v16i16_wmma_native, b.to_float());
            const v8i16_wmma_native  ci = __builtin_bit_cast(v8i16_wmma_native, c.to_float());
            const v8i16_wmma_native  di
                = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w64(ai, bi, ci, Opsel);
            return Value<v8bf_native, S, C>(__builtin_bit_cast(v8bf_native, di));
        }
        else
        {
            return detail::wmma_gfx11_half_16x16x16_dataflow<Opsel, true>(a, b, c);
        }
    }
#endif

#endif // defined(__GFX11__)

// =============================================================================
// RDNA4 (gfx1200/gfx1201) wave32 WMMA wrappers. Each wrapper is one
// macro-instantiation: type signature + a Float-mode call to the real builtin +
// FPSan-mode dispatch to the shared software dataflow. AMD's instruction-name
// convention: "fp8" = OCP E4M3FN, "bf8" = OCP E5M2.
//
// Gated on capability, not arch name: these builtins need the gfx12
// wmma-128b-insts + wavefrontsize32 features, which RDNA4 (gfx1200/gfx1201)
// has but gfx1250 does NOT (gfx1250 has a different WMMA family with K=32/64/128
// shapes -- see below). __has_builtin reflects exact per-target availability, so
// this block self-disables on gfx1250 and on CDNA (gfx9 family, including
// gfx950) where the matrix path is MFMA (fpsan/amdgcn_mfma.hpp, wave64 ABI). All
// eight 16x16x16 variants share the one feature, so one representative gate
// covers the whole block.
// =============================================================================
#if !defined(__HIP_DEVICE_COMPILE__) \
    || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)

// Generic wrapper: passes the native fragment to the builtin as-is. Used for
// f16/bf16 variants where Clang's ext_vector_type IS the builtin ABI.
#define FPSAN_DEFINE_WMMA_16X16X16(NAME, AVec_, BVec_, CVec_, BUILTIN)    \
    template <Semantics S, Conversions C>                                 \
    FPSAN_DEVICE Value<CVec_, S, C> NAME(                                 \
        Value<AVec_, S, C> a, Value<BVec_, S, C> b, Value<CVec_, S, C> c) \
    {                                                                     \
        if constexpr(S == Semantics::Native)                              \
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
        if constexpr(S == Semantics::Native)                                       \
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

#endif // __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)

#if !defined(__HIP_DEVICE_COMPILE__) \
    || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12)

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v4f_native, S, C> amdgcn_wmma_f32_16x16x16_f16_w64(Value<v4h_native, S, C> a,
                                                                          Value<v4h_native, S, C> b,
                                                                          Value<v4f_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v4f_native d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12(
                a.to_float(), b.to_float(), c.to_float());
            return Value<v4f_native, S, C>(d);
        }
        else
        {
            return detail::wmma_16x16x16_dataflow(a, b, c);
        }
    }

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v4f_native, S, C> amdgcn_wmma_f32_16x16x16_bf16_w64(
        Value<v4bf_native, S, C> a, Value<v4bf_native, S, C> b, Value<v4f_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            const v4i16_native ai = __builtin_bit_cast(v4i16_native, a.to_float());
            const v4i16_native bi = __builtin_bit_cast(v4i16_native, b.to_float());
            v4f_native d = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12(ai, bi, c.to_float());
            return Value<v4f_native, S, C>(d);
        }
        else
        {
            return detail::wmma_16x16x16_dataflow(a, b, c);
        }
    }

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v4h_native, S, C> amdgcn_wmma_f16_16x16x16_f16_w64(Value<v4h_native, S, C> a,
                                                                          Value<v4h_native, S, C> b,
                                                                          Value<v4h_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v4h_native d = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64_gfx12(
                a.to_float(), b.to_float(), c.to_float());
            return Value<v4h_native, S, C>(d);
        }
        else
        {
            return detail::wmma_16x16x16_dataflow(a, b, c);
        }
    }

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v4bf_native, S, C> amdgcn_wmma_bf16_16x16x16_bf16_w64(
        Value<v4bf_native, S, C> a, Value<v4bf_native, S, C> b, Value<v4bf_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            const v4i16_native ai = __builtin_bit_cast(v4i16_native, a.to_float());
            const v4i16_native bi = __builtin_bit_cast(v4i16_native, b.to_float());
            const v4i16_native ci = __builtin_bit_cast(v4i16_native, c.to_float());
            const v4i16_native di = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64_gfx12(ai, bi, ci);
            return Value<v4bf_native, S, C>(__builtin_bit_cast(v4bf_native, di));
        }
        else
        {
            return detail::wmma_16x16x16_dataflow(a, b, c);
        }
    }

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v4f_native, S, C> amdgcn_wmma_f32_16x16x16_fp8_fp8_w64(
        Value<v4e4m3_native, S, C> a, Value<v4e4m3_native, S, C> b, Value<v4f_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v4f_native d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w64_gfx12(
                __builtin_bit_cast(int, a.to_float()),
                __builtin_bit_cast(int, b.to_float()),
                c.to_float());
            return Value<v4f_native, S, C>(d);
        }
        else
        {
            return detail::wmma_16x16x16_dataflow(a, b, c);
        }
    }

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v4f_native, S, C> amdgcn_wmma_f32_16x16x16_fp8_bf8_w64(
        Value<v4e4m3_native, S, C> a, Value<v4e5m2_native, S, C> b, Value<v4f_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v4f_native d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w64_gfx12(
                __builtin_bit_cast(int, a.to_float()),
                __builtin_bit_cast(int, b.to_float()),
                c.to_float());
            return Value<v4f_native, S, C>(d);
        }
        else
        {
            return detail::wmma_16x16x16_dataflow(a, b, c);
        }
    }

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v4f_native, S, C> amdgcn_wmma_f32_16x16x16_bf8_fp8_w64(
        Value<v4e5m2_native, S, C> a, Value<v4e4m3_native, S, C> b, Value<v4f_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v4f_native d = __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w64_gfx12(
                __builtin_bit_cast(int, a.to_float()),
                __builtin_bit_cast(int, b.to_float()),
                c.to_float());
            return Value<v4f_native, S, C>(d);
        }
        else
        {
            return detail::wmma_16x16x16_dataflow(a, b, c);
        }
    }

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v4f_native, S, C> amdgcn_wmma_f32_16x16x16_bf8_bf8_w64(
        Value<v4e5m2_native, S, C> a, Value<v4e5m2_native, S, C> b, Value<v4f_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v4f_native d = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w64_gfx12(
                __builtin_bit_cast(int, a.to_float()),
                __builtin_bit_cast(int, b.to_float()),
                c.to_float());
            return Value<v4f_native, S, C>(d);
        }
        else
        {
            return detail::wmma_16x16x16_dataflow(a, b, c);
        }
    }

#endif // __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12)

// =============================================================================
// gfx1250 wave32 16x16x32 WMMA wrappers (K=32, 16-bit operands: f16 / bf16).
//
// Builtin ABI (validated against LLVM IR + Sema): the leading bool before A and
// before B are reserved-zero for f16/bf16; the i16 is the C accumulator modifier
// (0 none / 1 neg / 2 abs / 3 neg(abs)); the trailing two bools are
// matrix_a_reuse / matrix_b_reuse performance hints with NO numerical effect.
// So the operation is the plain D = A*B + C, which is exactly what the shared
// wmma_16x16x32_dataflow computes (C_mod=0, reuse=0). Gated on the gfx1250-only
// builtin, which is absent on RDNA4.
// =============================================================================
// =============================================================================
// gfx1250 wave32 16x16x4 WMMA wrapper (K=4, f32 operands).
//
// Builtin ABI mirrors the other gfx1250 WMMA shapes: a leading reserved-zero
// bool before A and before B, an i16 C-accumulator modifier (0 = none), the v8f
// C accumulator, and two trailing matrix_a/b_reuse perf-hint bools (no numeric
// effect). So the op is the plain D = A*B + C, which is exactly what the shared
// wmma_16x16x4_dataflow computes. gfx1250-only builtin (absent on RDNA4).
// =============================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x4_f32)
    template <Semantics S, Conversions C, int Cmod = 0>
    FPSAN_DEVICE Value<v8f_native, S, C> amdgcn_wmma_f32_16x16x4_f32(Value<v2f_native, S, C> a,
                                                                     Value<v2f_native, S, C> b,
                                                                     Value<v8f_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v8f_native d = __builtin_amdgcn_wmma_f32_16x16x4_f32(false,
                                                                 a.to_float(),
                                                                 false,
                                                                 b.to_float(),
                                                                 static_cast<short>(Cmod),
                                                                 c.to_float(),
                                                                 false,
                                                                 false);
            return Value<v8f_native, S, C>(d);
        }
        else
        {
            return detail::wmma_16x16x4_dataflow(a, b, detail::wmma_apply_cmod<Cmod>(c));
        }
    }
#endif // __has_builtin(__builtin_amdgcn_wmma_f32_16x16x4_f32)

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x32_f16)

#define FPSAN_DEFINE_WMMA_16X16X32(NAME, AVec_, CVec_, BUILTIN)                            \
    template <Semantics S, Conversions C, int Cmod = 0>                                    \
    FPSAN_DEVICE Value<CVec_, S, C> NAME(                                                  \
        Value<AVec_, S, C> a, Value<AVec_, S, C> b, Value<CVec_, S, C> c)                  \
    {                                                                                      \
        if constexpr(S == Semantics::Native)                                               \
        {                                                                                  \
            CVec_ d = BUILTIN(false,                                                       \
                              a.to_float(),                                                \
                              false,                                                       \
                              b.to_float(),                                                \
                              static_cast<short>(Cmod),                                    \
                              c.to_float(),                                                \
                              false,                                                       \
                              false);                                                      \
            return Value<CVec_, S, C>(d);                                                  \
        }                                                                                  \
        else                                                                               \
        {                                                                                  \
            return detail::wmma_16x16x32_dataflow(a, b, detail::wmma_apply_cmod<Cmod>(c)); \
        }                                                                                  \
    }

    FPSAN_DEFINE_WMMA_16X16X32(amdgcn_wmma_f32_16x16x32_f16,
                               v16h_native,
                               v8f_native,
                               __builtin_amdgcn_wmma_f32_16x16x32_f16)
    FPSAN_DEFINE_WMMA_16X16X32(amdgcn_wmma_f16_16x16x32_f16,
                               v16h_native,
                               v8h_native,
                               __builtin_amdgcn_wmma_f16_16x16x32_f16)
    FPSAN_DEFINE_WMMA_16X16X32(amdgcn_wmma_f32_16x16x32_bf16,
                               v16bf_native,
                               v8f_native,
                               __builtin_amdgcn_wmma_f32_16x16x32_bf16)
    FPSAN_DEFINE_WMMA_16X16X32(amdgcn_wmma_bf16_16x16x32_bf16,
                               v16bf_native,
                               v8bf_native,
                               __builtin_amdgcn_wmma_bf16_16x16x32_bf16)

    // Mixed C/D: f32 accumulator in, bf16 fragment out (f32 accumulate, narrow
    // once). Separate definition because C (v8f) and D (v8bf) differ in type.
    template <Semantics S, Conversions C, int Cmod = 0>
    FPSAN_DEVICE Value<v8bf_native, S, C> amdgcn_wmma_bf16f32_16x16x32_bf16(
        Value<v16bf_native, S, C> a, Value<v16bf_native, S, C> b, Value<v8f_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v8bf_native d = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(false,
                                                                        a.to_float(),
                                                                        false,
                                                                        b.to_float(),
                                                                        static_cast<short>(Cmod),
                                                                        c.to_float(),
                                                                        false,
                                                                        false);
            return Value<v8bf_native, S, C>(d);
        }
        else
        {
            return detail::
                wmma_16x16x32_dataflow_cdiff<v8bf_native, v16bf_native, v16bf_native, v8f_native>(
                    a, b, detail::wmma_apply_cmod<Cmod>(c));
        }
    }

#undef FPSAN_DEFINE_WMMA_16X16X32

#endif // __has_builtin(__builtin_amdgcn_wmma_f32_16x16x32_f16)

// =============================================================================
// gfx1250 wave32 16x16x64 fp8/bf8 WMMA wrappers (K=64, 8-bit operands).
//
// ABI: no leading neg bools (fp8 has no operand-neg modifier); the i16 is the C
// neg/abs modifier and the trailing two bools are reuse perf-hints -- so the op
// is the plain D = A*B + C with f32/f16 accumulation. AMD naming: 'fp8' = OCP
// E4M3FN, 'bf8' = OCP E5M2. The 32-fp8 per-lane fragment bit-casts to the v8i32
// builtin ABI. C/D are same-type (f32->f32 or f16->f16).
// =============================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8)

#define FPSAN_DEFINE_WMMA_16X16X64_FP8(NAME, AVec_, BVec_, CVec_, BUILTIN)                 \
    template <Semantics S, Conversions C, int Cmod = 0>                                    \
    FPSAN_DEVICE Value<CVec_, S, C> NAME(                                                  \
        Value<AVec_, S, C> a, Value<BVec_, S, C> b, Value<CVec_, S, C> c)                  \
    {                                                                                      \
        if constexpr(S == Semantics::Native)                                               \
        {                                                                                  \
            CVec_ d = BUILTIN(__builtin_bit_cast(v8i32_native, a.to_float()),              \
                              __builtin_bit_cast(v8i32_native, b.to_float()),              \
                              static_cast<short>(Cmod),                                    \
                              c.to_float(),                                                \
                              false,                                                       \
                              false);                                                      \
            return Value<CVec_, S, C>(d);                                                  \
        }                                                                                  \
        else                                                                               \
        {                                                                                  \
            return detail::wmma_16x16x64_dataflow(a, b, detail::wmma_apply_cmod<Cmod>(c)); \
        }                                                                                  \
    }

    FPSAN_DEFINE_WMMA_16X16X64_FP8(amdgcn_wmma_f32_16x16x64_fp8_fp8,
                                   v32e4m3_native,
                                   v32e4m3_native,
                                   v8f_native,
                                   __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8)
    FPSAN_DEFINE_WMMA_16X16X64_FP8(amdgcn_wmma_f32_16x16x64_fp8_bf8,
                                   v32e4m3_native,
                                   v32e5m2_native,
                                   v8f_native,
                                   __builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8)
    FPSAN_DEFINE_WMMA_16X16X64_FP8(amdgcn_wmma_f32_16x16x64_bf8_fp8,
                                   v32e5m2_native,
                                   v32e4m3_native,
                                   v8f_native,
                                   __builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8)
    FPSAN_DEFINE_WMMA_16X16X64_FP8(amdgcn_wmma_f32_16x16x64_bf8_bf8,
                                   v32e5m2_native,
                                   v32e5m2_native,
                                   v8f_native,
                                   __builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8)

    FPSAN_DEFINE_WMMA_16X16X64_FP8(amdgcn_wmma_f16_16x16x64_fp8_fp8,
                                   v32e4m3_native,
                                   v32e4m3_native,
                                   v8h_native,
                                   __builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8)
    FPSAN_DEFINE_WMMA_16X16X64_FP8(amdgcn_wmma_f16_16x16x64_fp8_bf8,
                                   v32e4m3_native,
                                   v32e5m2_native,
                                   v8h_native,
                                   __builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8)
    FPSAN_DEFINE_WMMA_16X16X64_FP8(amdgcn_wmma_f16_16x16x64_bf8_fp8,
                                   v32e5m2_native,
                                   v32e4m3_native,
                                   v8h_native,
                                   __builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8)
    FPSAN_DEFINE_WMMA_16X16X64_FP8(amdgcn_wmma_f16_16x16x64_bf8_bf8,
                                   v32e5m2_native,
                                   v32e5m2_native,
                                   v8h_native,
                                   __builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8)

#undef FPSAN_DEFINE_WMMA_16X16X64_FP8

#endif // __has_builtin(__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8)

// =============================================================================
// gfx1250 wave32 16x16x128 fp8/bf8 WMMA wrappers (K=128, 8-bit operands).
// Same ABI/semantics as the K=64 fp8 family (plain D = A*B + C); 64-fp8 per-lane
// fragment bit-casts to the v16i32 builtin ABI; C/D same-type (f32 or f16).
// =============================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8)

#define FPSAN_DEFINE_WMMA_16X16X128_FP8(NAME, AVec_, BVec_, CVec_, BUILTIN)                 \
    template <Semantics S, Conversions C, int Cmod = 0>                                     \
    FPSAN_DEVICE Value<CVec_, S, C> NAME(                                                   \
        Value<AVec_, S, C> a, Value<BVec_, S, C> b, Value<CVec_, S, C> c)                   \
    {                                                                                       \
        if constexpr(S == Semantics::Native)                                                \
        {                                                                                   \
            CVec_ d = BUILTIN(__builtin_bit_cast(v16i32_native, a.to_float()),              \
                              __builtin_bit_cast(v16i32_native, b.to_float()),              \
                              static_cast<short>(Cmod),                                     \
                              c.to_float(),                                                 \
                              false,                                                        \
                              false);                                                       \
            return Value<CVec_, S, C>(d);                                                   \
        }                                                                                   \
        else                                                                                \
        {                                                                                   \
            return detail::wmma_16x16x128_dataflow(a, b, detail::wmma_apply_cmod<Cmod>(c)); \
        }                                                                                   \
    }

    FPSAN_DEFINE_WMMA_16X16X128_FP8(amdgcn_wmma_f32_16x16x128_fp8_fp8,
                                    v64e4m3_native,
                                    v64e4m3_native,
                                    v8f_native,
                                    __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8)
    FPSAN_DEFINE_WMMA_16X16X128_FP8(amdgcn_wmma_f32_16x16x128_fp8_bf8,
                                    v64e4m3_native,
                                    v64e5m2_native,
                                    v8f_native,
                                    __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8)
    FPSAN_DEFINE_WMMA_16X16X128_FP8(amdgcn_wmma_f32_16x16x128_bf8_fp8,
                                    v64e5m2_native,
                                    v64e4m3_native,
                                    v8f_native,
                                    __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8)
    FPSAN_DEFINE_WMMA_16X16X128_FP8(amdgcn_wmma_f32_16x16x128_bf8_bf8,
                                    v64e5m2_native,
                                    v64e5m2_native,
                                    v8f_native,
                                    __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8)

    FPSAN_DEFINE_WMMA_16X16X128_FP8(amdgcn_wmma_f16_16x16x128_fp8_fp8,
                                    v64e4m3_native,
                                    v64e4m3_native,
                                    v8h_native,
                                    __builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8)
    FPSAN_DEFINE_WMMA_16X16X128_FP8(amdgcn_wmma_f16_16x16x128_fp8_bf8,
                                    v64e4m3_native,
                                    v64e5m2_native,
                                    v8h_native,
                                    __builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8)
    FPSAN_DEFINE_WMMA_16X16X128_FP8(amdgcn_wmma_f16_16x16x128_bf8_fp8,
                                    v64e5m2_native,
                                    v64e4m3_native,
                                    v8h_native,
                                    __builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8)
    FPSAN_DEFINE_WMMA_16X16X128_FP8(amdgcn_wmma_f16_16x16x128_bf8_bf8,
                                    v64e5m2_native,
                                    v64e5m2_native,
                                    v8h_native,
                                    __builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8)

#undef FPSAN_DEFINE_WMMA_16X16X128_FP8

#endif // __has_builtin(__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8)

// =============================================================================
// gfx1250 wave32 16x16x128 f8f6f4 WMMA wrapper (K=128, format-selectable A/B).
//
// One builtin whose A and B each carry a FORMAT immediate selecting how the
// fixed 512-bit (v16i32) per-lane fragment is interpreted: FP8(0)/BF8(1) 8-bit,
// FP6(2)/BF6(3) 6-bit, FP4(4) 4-bit. This `_sub` wrapper covers the sub-byte
// formats (fp6/bf6/fp4), which the dedicated 8-bit fp8/bf8 builtins above cannot
// reach. A and B may be different sub-byte widths (e.g. fp6 x fp4) -- all
// sub-byte formats share one physical k order. The operands are the raw packed
// v16i32 registers (sub-byte data is never a Value scalar element type): in Float
// mode their bits are the hardware codes; in FPSan mode they are the per-slot
// payloads packed at bit Width*s (see wmma_16x16x128_sub_dataflow). The op is the
// plain D = A*B + C (the i16 is the C neg/abs modifier, pinned 0). Mixing an
// 8-bit operand WITH a sub-byte one uses a different physical k order on hardware
// and is intentionally rejected here (static_assert) pending its own mix model.
// =============================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4)
    template <int         AFMT,
              int         BFMT,
              Semantics   S    = Semantics::Native,
              Conversions C    = Conversions::Explicit,
              int         Cmod = 0>
    FPSAN_DEVICE Value<v8f_native, S, C> amdgcn_wmma_f32_16x16x128_f8f6f4_sub(
        v16i32_native a, v16i32_native b, Value<v8f_native, S, C> c)
    {
        static_assert(AFMT >= 2 && AFMT <= 4 && BFMT >= 2 && BFMT <= 4,
                      "the _sub wrapper is for fp6/bf6/fp4 formats (2-4); use the dedicated "
                      "fp8/bf8 K=128 wrappers for 8-bit operands. Mixing an 8-bit operand with "
                      "a sub-byte one needs a dedicated WMMA mix model (not yet grounded).");
        if constexpr(S == Semantics::Native)
        {
            v8f_native d = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
                AFMT, a, BFMT, b, static_cast<short>(Cmod), c.to_float());
            return Value<v8f_native, S, C>(d);
        }
        else
        {
            int aw[16], bw[16];
            for(int w = 0; w < 16; ++w)
            {
                aw[w] = a[w];
                bw[w] = b[w];
            }
            return detail::wmma_16x16x128_sub_dataflow<detail::f8f6f4_width(AFMT),
                                                       detail::f8f6f4_width(BFMT),
                                                       S,
                                                       C>(aw, bw, detail::wmma_apply_cmod<Cmod>(c));
        }
    }

    // ---- gfx1250 wmma_f32_16x16x128_f8f6f4 MIXED 8-bit x sub-byte --------------
    // Pairs an 8-bit operand (fp8/bf8, fmt 0/1) with a sub-byte operand (fp6/bf6/
    // fp4, fmt 2-4). The two width classes use different physical k orders in the
    // per-lane fragment, so the FPSan dataflow stages the 8-bit operand on the
    // validated Wmma16x16x128Layout and the sub-byte operand on the mixed order
    // (detail::wmma_mix_sub_lane/slot, reverse-engineered from one-hot probes). Both
    // operands are the raw packed v16i32 registers. Float mode calls the builtin
    // (layout-agnostic). Exactly one operand must be 8-bit and the other sub-byte;
    // for same-class pairs use the `_sub` wrapper (sub x sub) or the dedicated
    // fp8/bf8 K=128 wrappers (8-bit x 8-bit). The i16 immediate is the C neg/abs
    // modifier (Cmod).
    template <int         AFMT,
              int         BFMT,
              Semantics   S    = Semantics::Native,
              Conversions C    = Conversions::Explicit,
              int         Cmod = 0>
    FPSAN_DEVICE Value<v8f_native, S, C> amdgcn_wmma_f32_16x16x128_f8f6f4_mixed(
        v16i32_native a, v16i32_native b, Value<v8f_native, S, C> c)
    {
        static_assert((AFMT <= 1) != (BFMT <= 1),
                      "the _mixed wrapper pairs exactly one 8-bit (fmt 0/1) with one sub-byte "
                      "(fmt 2-4) operand; use _sub for sub x sub or the fp8/bf8 wrappers for 8x8.");
        static_assert(AFMT >= 0 && AFMT <= 4 && BFMT >= 0 && BFMT <= 4,
                      "f8f6f4 format must be 0-4");
        if constexpr(S == Semantics::Native)
        {
            v8f_native d = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
                AFMT, a, BFMT, b, static_cast<short>(Cmod), c.to_float());
            return Value<v8f_native, S, C>(d);
        }
        else
        {
            int aw[16], bw[16];
            for(int w = 0; w < 16; ++w)
            {
                aw[w] = a[w];
                bw[w] = b[w];
            }
            return detail::wmma_16x16x128_mixed_dataflow<detail::f8f6f4_width(AFMT),
                                                         detail::f8f6f4_width(BFMT),
                                                         S,
                                                         C>(
                aw, bw, detail::wmma_apply_cmod<Cmod>(c));
        }
    }
#endif // __has_builtin(__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4)

// =============================================================================
// gfx1250 wave32 16x16x128 f8f6f4 BLOCK-SCALED WMMA wrapper (sub-byte operands).
//
// Per-operand block scale: K=128 in 4 K-blocks of 32, each (row, block) of A and
// (col, block) of B carries an E8M0 exponent; block kb's dot product is scaled by
// 2^(eA-127) * 2^(eB-127). `sa`/`sb` are the per-lane i32 scale operands; the
// scale_sel / scale_fmt immediates are pinned 0 (E8M0, natural byte=block
// mapping), the validated layout. Float mode forwards to the builtin (exact);
// FPSan mode runs the block-scaled sub-byte dataflow. As with the unscaled
// wrapper this covers sub-byte (fp6/bf6/fp4) operands; mixing an 8-bit with a
// sub-byte operand needs the (not-yet-grounded) WMMA mix model.
// =============================================================================
#if !defined(__HIP_DEVICE_COMPILE__) \
    || __has_builtin(__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4)
    template <int         AFMT,
              int         BFMT,
              Semantics   S         = Semantics::Native,
              Conversions C         = Conversions::Explicit,
              int         Cmod      = 0,
              int         AScaleFmt = 0,
              int         BScaleFmt = 0>
    FPSAN_DEVICE Value<v8f_native, S, C> amdgcn_wmma_scale_f32_16x16x128_f8f6f4_sub(
        v16i32_native a, v16i32_native b, Value<v8f_native, S, C> c, int sa, int sb)
    {
        static_assert(AFMT >= 2 && AFMT <= 4 && BFMT >= 2 && BFMT <= 4,
                      "the _sub wrapper is for fp6/bf6/fp4 formats (2-4); mixing an 8-bit operand "
                      "with a sub-byte one needs a dedicated WMMA mix model (not yet grounded).");
        if constexpr(S == Semantics::Native)
        {
            v8f_native d
                = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(AFMT,
                                                                   a,
                                                                   BFMT,
                                                                   b,
                                                                   static_cast<short>(Cmod),
                                                                   c.to_float(),
                                                                   0,
                                                                   AScaleFmt,
                                                                   sa,
                                                                   0,
                                                                   BScaleFmt,
                                                                   sb,
                                                                   false,
                                                                   false);
            return Value<v8f_native, S, C>(d);
        }
        else
        {
            int aw[16], bw[16];
            for(int w = 0; w < 16; ++w)
            {
                aw[w] = a[w];
                bw[w] = b[w];
            }
            return detail::wmma_16x16x128_sub_scaled_dataflow<detail::f8f6f4_width(AFMT),
                                                              detail::f8f6f4_width(BFMT),
                                                              S,
                                                              C,
                                                              AScaleFmt,
                                                              BScaleFmt>(
                aw, bw, detail::wmma_apply_cmod<Cmod>(c), sa, sb);
        }
    }

    // ---- gfx1250 wmma_scale_f32_16x16x128_f8f6f4 MIXED 8-bit x sub-byte ---------
    // Block-scaled mix: pairs one 8-bit (fmt 0/1) with one sub-byte (fmt 2-4)
    // operand. FPSan stages operands on the mixed layout and applies the mixed
    // scale-byte map (both operands use wmma_mix_scale_byte). Float forwards to
    // the builtin. AScaleFmt/BScaleFmt: 0 = E8M0FNU, 2 = E4M3FN.
    template <int         AFMT,
              int         BFMT,
              Semantics   S         = Semantics::Native,
              Conversions C         = Conversions::Explicit,
              int         Cmod      = 0,
              int         AScaleFmt = 0,
              int         BScaleFmt = 0>
    FPSAN_DEVICE Value<v8f_native, S, C> amdgcn_wmma_scale_f32_16x16x128_f8f6f4_mixed(
        v16i32_native a, v16i32_native b, Value<v8f_native, S, C> c, int sa, int sb)
    {
        static_assert((AFMT <= 1) != (BFMT <= 1),
                      "the _mixed scaled wrapper pairs exactly one 8-bit (fmt 0/1) with one "
                      "sub-byte (fmt 2-4) operand; use _sub for sub x sub.");
        static_assert(AFMT >= 0 && AFMT <= 4 && BFMT >= 0 && BFMT <= 4,
                      "f8f6f4 format must be 0-4");
        if constexpr(S == Semantics::Native)
        {
            v8f_native d
                = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(AFMT,
                                                                   a,
                                                                   BFMT,
                                                                   b,
                                                                   static_cast<short>(Cmod),
                                                                   c.to_float(),
                                                                   0,
                                                                   AScaleFmt,
                                                                   sa,
                                                                   0,
                                                                   BScaleFmt,
                                                                   sb,
                                                                   false,
                                                                   false);
            return Value<v8f_native, S, C>(d);
        }
        else
        {
            int aw[16], bw[16];
            for(int w = 0; w < 16; ++w)
            {
                aw[w] = a[w];
                bw[w] = b[w];
            }
            return detail::wmma_16x16x128_mixed_scaled_dataflow<detail::f8f6f4_width(AFMT),
                                                                detail::f8f6f4_width(BFMT),
                                                                S,
                                                                C,
                                                                AScaleFmt,
                                                                BScaleFmt>(
                aw, bw, detail::wmma_apply_cmod<Cmod>(c), sa, sb);
        }
    }
#endif // __has_builtin(__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4)

// =============================================================================
// gfx1250 wave32 16x16x128 f8f6f4 scale16 (i64) BLOCK-SCALED WMMA wrapper
// (sub-byte operands). Same per-operand E8M0 block scaling as the i32 variant,
// but each scale operand is a 64-bit word carrying 8 E8M0 bytes (finer
// granularity); the controlling byte is detail::wmma_sub_scale16_byte(k). Scale
// sel/fmt immediates pinned 0 (E8M0). Float mode forwards to the builtin (exact);
// FPSan mode runs the scale16 sub-byte dataflow. Sub-byte (fp6/bf6/fp4) only.
// =============================================================================
#if !defined(__HIP_DEVICE_COMPILE__) \
    || __has_builtin(__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4)
    template <int         AFMT,
              int         BFMT,
              Semantics   S         = Semantics::Native,
              Conversions C         = Conversions::Explicit,
              int         Cmod      = 0,
              int         AScaleFmt = 0,
              int         BScaleFmt = 0>
    FPSAN_DEVICE Value<v8f_native, S, C> amdgcn_wmma_scale16_f32_16x16x128_f8f6f4_sub(
        v16i32_native a, v16i32_native b, Value<v8f_native, S, C> c, long sa, long sb)
    {
        static_assert(AFMT >= 2 && AFMT <= 4 && BFMT >= 2 && BFMT <= 4,
                      "the _sub wrapper is for fp6/bf6/fp4 formats (2-4); mixing an 8-bit operand "
                      "with a sub-byte one needs a dedicated WMMA mix model (not yet grounded).");
        if constexpr(S == Semantics::Native)
        {
            v8f_native d
                = __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(AFMT,
                                                                     a,
                                                                     BFMT,
                                                                     b,
                                                                     static_cast<short>(Cmod),
                                                                     c.to_float(),
                                                                     0,
                                                                     AScaleFmt,
                                                                     sa,
                                                                     0,
                                                                     BScaleFmt,
                                                                     sb,
                                                                     false,
                                                                     false);
            return Value<v8f_native, S, C>(d);
        }
        else
        {
            int aw[16], bw[16];
            for(int w = 0; w < 16; ++w)
            {
                aw[w] = a[w];
                bw[w] = b[w];
            }
            return detail::wmma_16x16x128_sub_scaled16_dataflow<detail::f8f6f4_width(AFMT),
                                                                detail::f8f6f4_width(BFMT),
                                                                S,
                                                                C,
                                                                AScaleFmt,
                                                                BScaleFmt>(
                aw,
                bw,
                detail::wmma_apply_cmod<Cmod>(c),
                static_cast<long long>(sa),
                static_cast<long long>(sb));
        }
    }

    // ---- gfx1250 wmma_scale16_f32_16x16x128_f8f6f4 MIXED 8-bit x sub-byte -------
    // scale16 (i64, 8-byte / finer) counterpart of the mixed scaled wrapper;
    // controlling byte = detail::wmma_mix_scale16_byte(k). Float forwards to the
    // builtin; FPSan runs the mixed scaled16 dataflow.
    template <int         AFMT,
              int         BFMT,
              Semantics   S         = Semantics::Native,
              Conversions C         = Conversions::Explicit,
              int         Cmod      = 0,
              int         AScaleFmt = 0,
              int         BScaleFmt = 0>
    FPSAN_DEVICE Value<v8f_native, S, C> amdgcn_wmma_scale16_f32_16x16x128_f8f6f4_mixed(
        v16i32_native a, v16i32_native b, Value<v8f_native, S, C> c, long sa, long sb)
    {
        static_assert((AFMT <= 1) != (BFMT <= 1),
                      "the _mixed scale16 wrapper pairs exactly one 8-bit (fmt 0/1) with one "
                      "sub-byte (fmt 2-4) operand; use _sub for sub x sub.");
        static_assert(AFMT >= 0 && AFMT <= 4 && BFMT >= 0 && BFMT <= 4,
                      "f8f6f4 format must be 0-4");
        if constexpr(S == Semantics::Native)
        {
            v8f_native d
                = __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(AFMT,
                                                                     a,
                                                                     BFMT,
                                                                     b,
                                                                     static_cast<short>(Cmod),
                                                                     c.to_float(),
                                                                     0,
                                                                     AScaleFmt,
                                                                     sa,
                                                                     0,
                                                                     BScaleFmt,
                                                                     sb,
                                                                     false,
                                                                     false);
            return Value<v8f_native, S, C>(d);
        }
        else
        {
            int aw[16], bw[16];
            for(int w = 0; w < 16; ++w)
            {
                aw[w] = a[w];
                bw[w] = b[w];
            }
            return detail::wmma_16x16x128_mixed_scaled16_dataflow<detail::f8f6f4_width(AFMT),
                                                                  detail::f8f6f4_width(BFMT),
                                                                  S,
                                                                  C,
                                                                  AScaleFmt,
                                                                  BScaleFmt>(
                aw,
                bw,
                detail::wmma_apply_cmod<Cmod>(c),
                static_cast<long long>(sa),
                static_cast<long long>(sb));
        }
    }
#endif // __has_builtin(__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4)

// =============================================================================
// gfx1250 wave32 32x16x128 f4 (E2M1) WMMA wrapper. A is 32x128 fp4 (v16i32), B
// is 128x16 fp4 (v8i32), C/D is 32x16 f32 (v16f32). Float mode forwards to the
// builtin (exact); FPSan mode runs the grounded 32x16x128 f4 dataflow. The
// fragment ABI is the silicon-grounded layout in detail::Wmma32x16x128F4Layout.
// =============================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f32_32x16x128_f4)
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit, int Cmod = 0>
    FPSAN_DEVICE Value<v16f_native, S, C>
        amdgcn_wmma_f32_32x16x128_f4(v16i32_native a, v8i32_native b, Value<v16f_native, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            v16f_native d = __builtin_amdgcn_wmma_f32_32x16x128_f4(
                a, b, static_cast<short>(Cmod), c.to_float());
            return Value<v16f_native, S, C>(d);
        }
        else
        {
            int aw[16], bw[16];
            for(int w = 0; w < 16; ++w)
                aw[w] = a[w];
            for(int w = 0; w < 16; ++w)
                bw[w] = (w < 8) ? b[w] : 0;
            return detail::wmma_32x16x128_f4_dataflow<S, C>(
                aw, bw, detail::wmma_apply_cmod<Cmod>(c));
        }
    }
#endif // __has_builtin(__builtin_amdgcn_wmma_f32_32x16x128_f4)

// =============================================================================
// gfx1250 wave32 32x16x128 f4 BLOCK-SCALED WMMA wrappers (i32 scale + i64
// scale16). Per-operand E8M0 block scale; sel/fmt immediates pinned 0 (E8M0).
// Float forwards to the builtin (exact); FPSan runs the grounded scaled dataflow
// (A-scale word from lane scale_lane_a(m), B-scale word from lane n; byte map
// detail::wmma_sub_scale_byte / wmma_sub_scale16_byte).
// =============================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_scale_f32_32x16x128_f4)
    template <Semantics   S         = Semantics::Native,
              Conversions C         = Conversions::Explicit,
              int         Cmod      = 0,
              int         AScaleFmt = 0,
              int         BScaleFmt = 0>
    FPSAN_DEVICE Value<v16f_native, S, C> amdgcn_wmma_scale_f32_32x16x128_f4(
        v16i32_native a, v8i32_native b, Value<v16f_native, S, C> c, int sa, int sb)
    {
        if constexpr(S == Semantics::Native)
        {
            v16f_native d = __builtin_amdgcn_wmma_scale_f32_32x16x128_f4(a,
                                                                         b,
                                                                         static_cast<short>(Cmod),
                                                                         c.to_float(),
                                                                         0,
                                                                         AScaleFmt,
                                                                         sa,
                                                                         0,
                                                                         BScaleFmt,
                                                                         sb,
                                                                         false,
                                                                         false);
            return Value<v16f_native, S, C>(d);
        }
        else
        {
            int aw[16], bw[16];
            for(int w = 0; w < 16; ++w)
                aw[w] = a[w];
            for(int w = 0; w < 16; ++w)
                bw[w] = (w < 8) ? b[w] : 0;
            return detail::wmma_32x16x128_f4_scaled_dataflow<S, C, AScaleFmt, BScaleFmt>(
                aw, bw, detail::wmma_apply_cmod<Cmod>(c), sa, sb);
        }
    }
#endif // __has_builtin(__builtin_amdgcn_wmma_scale_f32_32x16x128_f4)

#if !defined(__HIP_DEVICE_COMPILE__) \
    || __has_builtin(__builtin_amdgcn_wmma_scale16_f32_32x16x128_f4)
    template <Semantics   S         = Semantics::Native,
              Conversions C         = Conversions::Explicit,
              int         Cmod      = 0,
              int         AScaleFmt = 0,
              int         BScaleFmt = 0>
    FPSAN_DEVICE Value<v16f_native, S, C> amdgcn_wmma_scale16_f32_32x16x128_f4(
        v16i32_native a, v8i32_native b, Value<v16f_native, S, C> c, long sa, long sb)
    {
        if constexpr(S == Semantics::Native)
        {
            v16f_native d = __builtin_amdgcn_wmma_scale16_f32_32x16x128_f4(a,
                                                                           b,
                                                                           static_cast<short>(Cmod),
                                                                           c.to_float(),
                                                                           0,
                                                                           AScaleFmt,
                                                                           sa,
                                                                           0,
                                                                           BScaleFmt,
                                                                           sb,
                                                                           false,
                                                                           false);
            return Value<v16f_native, S, C>(d);
        }
        else
        {
            int aw[16], bw[16];
            for(int w = 0; w < 16; ++w)
                aw[w] = a[w];
            for(int w = 0; w < 16; ++w)
                bw[w] = (w < 8) ? b[w] : 0;
            return detail::wmma_32x16x128_f4_scaled16_dataflow<S, C, AScaleFmt, BScaleFmt>(
                aw,
                bw,
                detail::wmma_apply_cmod<Cmod>(c),
                static_cast<long long>(sa),
                static_cast<long long>(sb));
        }
    }
#endif // __has_builtin(__builtin_amdgcn_wmma_scale16_f32_32x16x128_f4)

} // namespace fpsan

#endif // FPSAN_AMDGCN_MATRIX_HPP
