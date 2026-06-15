// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/amdgcn_cvt.hpp
// ----------------------------------------------------------------------------
// FPSan wrappers for AMDGPU conversion/packing intrinsics. Opt-in (not pulled
// by <fpsan/fpsan.hpp>).
//
// Implemented: amdgcn_cvt_pkrtz; cvt_f32_{fp8,bf8}; cvt_pk_{fp8,bf8}_f32;
// cvt_pk_f32_{fp8,bf8}; cvt_sr_{fp8,bf8,f16,bf16}_f32; the fp8/bf8 scaled
// scalar/pack pieces; and the full MX sub-byte scaled family:
//   - fp4 (e2m1): cvt_scalef32_pk_{f32,f16,bf16}_fp4 (unpack),
//     cvt_scalef32_pk_fp4_{f32,f16,bf16} + sr_pk_fp4_f32 (pack).
//   - fp6 (e2m3) / bf6 (e3m2): cvt_scalef32_pk32_{f32,f16,bf16}_{fp6,bf6}
//     (unpack), cvt_scalef32_2xpk16_{fp6,bf6}_f32 (pack, interleaved),
//     cvt_scalef32_pk32_{fp6,bf6}_{f16,bf16} (pack, contiguous), and
//     cvt_scalef32_sr_pk32_{fp6,bf6}_f32 (SR pack).
// Sub-byte data never becomes a Value element type (fp4/fp6 are not byte
// multiples); it lives packed in a u32 (8 fp4 nibbles) or v6u32 (32 fp6 codes,
// 192-bit stream), and FPSan treats those as packed PAYLOADS. Every
// gfx950-selectable cvt_scalef32 op is now wrapped + golden-tested; only the
// gfx1250-only pk8/pk16 wide forms remain (not selectable on gfx950).
// ----------------------------------------------------------------------------
#ifndef FPSAN_AMDGCN_CVT_HPP
#define FPSAN_AMDGCN_CVT_HPP

#include "fpsan/cast.hpp"
#include "fpsan/detail/fp8.hpp"
#include "fpsan/detail/native_vec.hpp"
#include "fpsan/detail/subbyte_widen.hpp"
#include "fpsan/value.hpp"

#include <cstdint>

#if !defined(__HIP__) && !defined(__CUDACC__)
#error "fpsan/amdgcn_cvt.hpp is GPU-only; compile as HIP (or CUDA)."
#endif

namespace fpsan
{

    // v2h_native / v2bf_native come from fpsan/detail/native_vec.hpp.
    using v2f_native = float __attribute__((ext_vector_type(2)));
    // The scaled pk_{fp8,bf8}_f32 builtins type their packed register as a
    // _Vector<2,short>; it is bit-identical to the 32-bit int our wrappers use, so
    // we bit_cast at the call boundary and keep the int public API.
    using v2s16_native = short __attribute__((ext_vector_type(2)));

    // Pack two f32 values into a v2f16 with round-to-zero. Float mode forwards to
    // the builtin; FPSan mode uses two scalar fpsan::cast<_Float16>(...) and
    // assembles them into the v2h fragment (cast does signed-truncate of the
    // payload 32->16, which is the FPSan model of an f32 -> f16 conversion).
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<v2h_native, S, C> amdgcn_cvt_pkrtz(Value<float, S, C> a,
                                                          Value<float, S, C> b)
    {
        if constexpr(S == Semantics::Native)
        {
            // The builtin returns ExtVector<2, __fp16>; bit_cast to our _Float16
            // vector (same storage, different Clang type spelling).
            auto       raw    = __builtin_amdgcn_cvt_pkrtz(a.to_float(), b.to_float());
            v2h_native packed = __builtin_bit_cast(v2h_native, raw);
            return Value<v2h_native, S, C>(packed);
        }
        else
        {
            auto                    ah = fpsan::cast<_Float16>(a);
            auto                    bh = fpsan::cast<_Float16>(b);
            Value<v2h_native, S, C> r{};
            r.set(0, ah);
            r.set(1, bh);
            return r;
        }
    }

    // =============================================================================
    // gfx1250 stochastic-rounding pack to f16 / bf16 (cvt_sr_pk_{f16,bf16}_f32;
    // "gfx1250-insts" / "bf16-cvt-insts").
    //
    // Pack two f32 into a v2{f16,bf16} with stochastic rounding from a single u32
    // seed. Unlike cvt_sr_f16_f32 there is no `old` register and no lane select:
    // both result lanes are produced (lane0 <- a, lane1 <- b). FPSan model is two
    // scalar fpsan::cast<{_Float16,__bf16}>(...) (payload signed-truncate 32->16,
    // the f32->f16/bf16 narrowing model); the seed is opaque to that deterministic
    // answer, exactly like every other SR wrapper here. Float mode forwards to the
    // builtin and the tests assert against the host narrowing at exactly
    // representable inputs (where SR is exact) plus a neighbor-bracket property for
    // non-representable inputs.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_sr_pk_f16_f32)
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<v2h_native, S, C>
        amdgcn_cvt_sr_pk_f16_f32(Value<float, S, C> a, Value<float, S, C> b, std::uint32_t seed)
    {
        if constexpr(S == Semantics::Native)
        {
            auto raw = __builtin_amdgcn_cvt_sr_pk_f16_f32(
                a.to_float(), b.to_float(), static_cast<int>(seed));
            return Value<v2h_native, S, C>(__builtin_bit_cast(v2h_native, raw));
        }
        else
        {
            (void)seed;
            Value<v2h_native, S, C> r{};
            r.set(0, fpsan::cast<_Float16>(a));
            r.set(1, fpsan::cast<_Float16>(b));
            return r;
        }
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_sr_pk_bf16_f32)
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<v2bf_native, S, C>
        amdgcn_cvt_sr_pk_bf16_f32(Value<float, S, C> a, Value<float, S, C> b, std::uint32_t seed)
    {
        if constexpr(S == Semantics::Native)
        {
            auto raw = __builtin_amdgcn_cvt_sr_pk_bf16_f32(
                a.to_float(), b.to_float(), static_cast<int>(seed));
            return Value<v2bf_native, S, C>(__builtin_bit_cast(v2bf_native, raw));
        }
        else
        {
            (void)seed;
            Value<v2bf_native, S, C> r{};
            r.set(0, fpsan::cast<__bf16>(a));
            r.set(1, fpsan::cast<__bf16>(b));
            return r;
        }
    }
#endif

    // =============================================================================
    // FP8 packed conversions (gfx12 fp8-conversion-insts).
    //
    // AMD's fp8 cvt builtins operate on packed int32 registers, each holding 4
    // fp8 (or bf8) bytes. The byte position is selected by a constexpr index.
    // Customers wire them up at the byte level, and the FPSan wrappers preserve
    // that: the int argument is opaque to FPSan (it's a raw register holding
    // either real fp8 bytes in Float mode or fp8 PAYLOAD bytes in FPSan mode,
    // the customer's choice consistent within their kernel).
    // =============================================================================

    namespace detail
    {
        // Helper: extract byte `Idx` from a packed int and rebuild a Value<FP8>.
        template <int Idx, class FP8, Semantics S, Conversions C>
        FPSAN_DEVICE Value<FP8, S, C> unpack_byte_to_fp8(int packed)
        {
            const std::uint32_t u    = static_cast<std::uint32_t>(packed);
            const std::uint8_t  byte = static_cast<std::uint8_t>((u >> (Idx * 8)) & 0xFFu);
            if constexpr(S == Semantics::Native)
                return Value<FP8, S, C>(FP8(byte));
            else
                return Value<FP8, S, C>::from_fpsan_payload(byte);
        }
        // Splice an FP8 byte at position 0 or 1 of the low/high halfword.
        template <bool DstLo>
        FPSAN_DEVICE int pack_fp8_pair(std::uint8_t a_byte, std::uint8_t b_byte, int old)
        {
            std::uint32_t u = static_cast<std::uint32_t>(old);
            if constexpr(DstLo)
            {
                u = (u & 0xFFFF0000u) | static_cast<std::uint32_t>(a_byte)
                    | (static_cast<std::uint32_t>(b_byte) << 8);
            }
            else
            {
                u = (u & 0x0000FFFFu) | (static_cast<std::uint32_t>(a_byte) << 16)
                    | (static_cast<std::uint32_t>(b_byte) << 24);
            }
            return static_cast<int>(u);
        }
    } // namespace detail

// cvt_f32_fp8 / cvt_f32_bf8: read byte ByteIdx of a packed int as an fp8 value
// and convert it to f32. Float mode forwards to the builtin; FPSan mode
// reinterprets the byte as an fp8 payload and uses fpsan::cast<float> (which
// is signed-extend 8 -> 32).
#define FPSAN_DEFINE_CVT_F32_FP8(name, FP8, BUILTIN)                                               \
    template <int ByteIdx, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE Value<float, S, C> name(int packed)                                               \
    {                                                                                              \
        if constexpr(S == Semantics::Native)                                                       \
            return Value<float, S, C>(BUILTIN(packed, ByteIdx));                                   \
        else                                                                                       \
            return fpsan::cast<float>(detail::unpack_byte_to_fp8<ByteIdx, FP8, S, C>(packed));     \
    }
    FPSAN_DEFINE_CVT_F32_FP8(amdgcn_cvt_f32_fp8, fp8_e4m3, __builtin_amdgcn_cvt_f32_fp8)
    FPSAN_DEFINE_CVT_F32_FP8(amdgcn_cvt_f32_bf8, fp8_e5m2, __builtin_amdgcn_cvt_f32_bf8)
#undef FPSAN_DEFINE_CVT_F32_FP8

// cvt_pk_fp8_f32 / cvt_pk_bf8_f32: convert two f32 values to fp8 and splice
// them into a packed int at the low (DstLo=true) or high (DstLo=false) half.
// `old` is the int register's pre-existing contents, returned unchanged in
// the bytes we don't touch. FPSan mode uses fpsan::cast<FP8> (which truncates
// the f32 payload 32 -> 8) and packs the resulting bytes.
#define FPSAN_DEFINE_CVT_PK_FP8(name, FP8, BUILTIN)                                               \
    template <bool DstLo, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE int name(Value<float, S, C> a, Value<float, S, C> b, int old)                    \
    {                                                                                             \
        if constexpr(S == Semantics::Native)                                                      \
            /* Builtin word-select is false=low, true=high (verified on gfx950);     \
         DstLo=true must write the LOW half, matching the FPSan path below, so \
         pass !DstLo. (Without this, Float and FPSan place the byte in         \
         different words -- a silent cross-mode inconsistency.) */            \
            return BUILTIN(a.to_float(), b.to_float(), old, !DstLo);                              \
        else                                                                                      \
        {                                                                                         \
            const auto af = fpsan::cast<FP8>(a);                                                  \
            const auto bf = fpsan::cast<FP8>(b);                                                  \
            return detail::pack_fp8_pair<DstLo>(static_cast<std::uint8_t>(af.fpsan_payload()),    \
                                                static_cast<std::uint8_t>(bf.fpsan_payload()),    \
                                                old);                                             \
        }                                                                                         \
    }
    FPSAN_DEFINE_CVT_PK_FP8(amdgcn_cvt_pk_fp8_f32, fp8_e4m3, __builtin_amdgcn_cvt_pk_fp8_f32)
    FPSAN_DEFINE_CVT_PK_FP8(amdgcn_cvt_pk_bf8_f32, fp8_e5m2, __builtin_amdgcn_cvt_pk_bf8_f32)
#undef FPSAN_DEFINE_CVT_PK_FP8

// cvt_pk_f32_fp8 / cvt_pk_f32_bf8: the unpack direction -- read the WordSel'th
// 16-bit word (two fp8 bytes 2*WordSel, 2*WordSel+1) of a packed int and expand
// it to two f32. FPSan mode reinterprets each byte as an fp8 payload and uses
// fpsan::cast<float>.
#define FPSAN_DEFINE_CVT_PK_F32_FP8(name, FP8, BUILTIN)                                            \
    template <bool        WordSel,                                                                 \
              Semantics   S = Semantics::Native,                                                   \
              Conversions C = Conversions::Explicit>                                               \
    FPSAN_DEVICE Value<v2f_native, S, C> name(int packed)                                          \
    {                                                                                              \
        if constexpr(S == Semantics::Native)                                                       \
        {                                                                                          \
            v2f_native r = BUILTIN(packed, WordSel);                                               \
            return Value<v2f_native, S, C>(r);                                                     \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            Value<v2f_native, S, C> r{};                                                           \
            r.set(0,                                                                               \
                  fpsan::cast<float>(detail::unpack_byte_to_fp8<2 * WordSel, FP8, S, C>(packed))); \
            r.set(1,                                                                               \
                  fpsan::cast<float>(                                                              \
                      detail::unpack_byte_to_fp8<2 * WordSel + 1, FP8, S, C>(packed)));            \
            return r;                                                                              \
        }                                                                                          \
    }
    FPSAN_DEFINE_CVT_PK_F32_FP8(amdgcn_cvt_pk_f32_fp8, fp8_e4m3, __builtin_amdgcn_cvt_pk_f32_fp8)
    FPSAN_DEFINE_CVT_PK_F32_FP8(amdgcn_cvt_pk_f32_bf8, fp8_e5m2, __builtin_amdgcn_cvt_pk_f32_bf8)
#undef FPSAN_DEFINE_CVT_PK_F32_FP8

    // =============================================================================
    // Stochastic-rounding cvt (fp8-conversion-insts; supported on gfx950 and
    // gfx12 alike).
    //
    // cvt_sr_{fp8,bf8}_f32(val, packed_old, seed, byte_idx)
    //   Takes a single f32 'val', an int 'packed_old' (the destination register's
    //   prior contents), an u32 'seed' for the stochastic rounding, and a 0..3
    //   constexpr byte index selecting which byte of `packed_old` the new fp8
    //   byte replaces.  Returns the updated packed int.
    //
    //   NOTE on builtin arg order: the wrapper API is (val, old, seed), but the
    //   clang builtin signature is (val, SEED, OLD, byte_sel) -- verified from
    //   clang/test/CodeGenOpenCL/builtins-amdgcn-fp8.cl and ...-gfx1250.cl, which
    //   lower __builtin_amdgcn_cvt_sr_fp8_f32(a, b, old, 3) to
    //   llvm.amdgcn.cvt.sr.fp8.f32(val, seed=b, old, 3). We therefore forward
    //   (val, seed, packed_old, ByteIdx). (The FPSan path always splices into
    //   packed_old, which is the destination register, so both paths agree that
    //   packed_old is `old`.)
    //
    // FPSan model: stochastic rounding is just rounding with an extra randomness
    // source.  Since FPSan cast<FP8> already truncates the payload deterministic-
    // ally (Triton ext/trunc), the seed parameter is opaque -- it doesn't change
    // the payload-domain answer.  We splice the cast<FP8>(val) payload into the
    // chosen byte of `packed_old`, matching the cvt_pk_fp8_f32 contract for the
    // chosen byte slot.
    // =============================================================================

    namespace detail
    {
        template <int ByteIdx>
        FPSAN_DEVICE int splice_fp8_byte(std::uint8_t byte, int old)
        {
            static_assert(ByteIdx >= 0 && ByteIdx < 4, "ByteIdx must be 0..3");
            const std::uint32_t mask = ~(std::uint32_t{0xFFu} << (ByteIdx * 8));
            std::uint32_t       u    = static_cast<std::uint32_t>(old) & mask;
            u |= static_cast<std::uint32_t>(byte) << (ByteIdx * 8);
            return static_cast<int>(u);
        }
    } // namespace detail

#define FPSAN_DEFINE_CVT_SR_FP8(NAME, FP8, BUILTIN)                                                \
    template <int ByteIdx, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE int NAME(Value<float, S, C> val, int packed_old, std::uint32_t seed)              \
    {                                                                                              \
        if constexpr(S == Semantics::Native)                                                       \
            return BUILTIN(val.to_float(), static_cast<int>(seed), packed_old, ByteIdx);           \
        else                                                                                       \
        {                                                                                          \
            (void)seed;                                                                            \
            const auto f8 = fpsan::cast<FP8>(val);                                                 \
            return detail::splice_fp8_byte<ByteIdx>(static_cast<std::uint8_t>(f8.fpsan_payload()), \
                                                    packed_old);                                   \
        }                                                                                          \
    }

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_sr_fp8_f32)
    FPSAN_DEFINE_CVT_SR_FP8(amdgcn_cvt_sr_fp8_f32, fp8_e4m3, __builtin_amdgcn_cvt_sr_fp8_f32)
    FPSAN_DEFINE_CVT_SR_FP8(amdgcn_cvt_sr_bf8_f32, fp8_e5m2, __builtin_amdgcn_cvt_sr_bf8_f32)
#endif
#undef FPSAN_DEFINE_CVT_SR_FP8

    // =============================================================================
    // gfx1250 f16 <-> fp8/bf8 conversions ("gfx1250-insts").
    //
    // gfx1250 adds direct f16<->fp8 paths (RDNA4/CDNA only had f32<->fp8). The
    // fp8 formats are the SAME OCP encodings as the f32 ops -- fp8 = e4m3, bf8 =
    // e5m2 -- so we reuse fp8_e4m3 / fp8_e5m2 and the existing payload-ring cast
    // (fpsan::cast<FP8> truncates, fpsan::cast<_Float16> resizes). Float mode
    // forwards to the builtin; FPSan mode is the deterministic Triton ext/trunc
    // cast, identical in spirit to the f32 family. Every fp8 value is exactly
    // representable in f16, so the decode direction is lossless. Each block is
    // __has_builtin-gated (gfx1250-exclusive).

    // cvt_f16_fp8 / cvt_f16_bf8: read byte ByteIdx of a packed int as fp8 -> f16.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_f16_fp8)
#define FPSAN_DEFINE_CVT_F16_FP8(name, FP8, BUILTIN)                                               \
    template <int ByteIdx, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE Value<_Float16, S, C> name(int packed)                                            \
    {                                                                                              \
        if constexpr(S == Semantics::Native)                                                       \
            return Value<_Float16, S, C>(BUILTIN(packed, ByteIdx));                                \
        else                                                                                       \
            return fpsan::cast<_Float16>(detail::unpack_byte_to_fp8<ByteIdx, FP8, S, C>(packed));  \
    }
    FPSAN_DEFINE_CVT_F16_FP8(amdgcn_cvt_f16_fp8, fp8_e4m3, __builtin_amdgcn_cvt_f16_fp8)
    FPSAN_DEFINE_CVT_F16_FP8(amdgcn_cvt_f16_bf8, fp8_e5m2, __builtin_amdgcn_cvt_f16_bf8)
#undef FPSAN_DEFINE_CVT_F16_FP8
#endif

    // cvt_pk_f16_fp8 / cvt_pk_f16_bf8: unpack the two fp8 bytes of a 16-bit short
    // to two f16. (No word-select: the short IS the packed pair.)
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_pk_f16_fp8)
#define FPSAN_DEFINE_CVT_PK_F16_FP8(name, FP8, BUILTIN)                                        \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>          \
    FPSAN_DEVICE Value<v2h_native, S, C> name(short packed)                                    \
    {                                                                                          \
        if constexpr(S == Semantics::Native)                                                   \
            return Value<v2h_native, S, C>(BUILTIN(packed));                                   \
        else                                                                                   \
        {                                                                                      \
            const int               p = static_cast<int>(static_cast<unsigned short>(packed)); \
            Value<v2h_native, S, C> r{};                                                       \
            r.set(0, fpsan::cast<_Float16>(detail::unpack_byte_to_fp8<0, FP8, S, C>(p)));      \
            r.set(1, fpsan::cast<_Float16>(detail::unpack_byte_to_fp8<1, FP8, S, C>(p)));      \
            return r;                                                                          \
        }                                                                                      \
    }
    FPSAN_DEFINE_CVT_PK_F16_FP8(amdgcn_cvt_pk_f16_fp8, fp8_e4m3, __builtin_amdgcn_cvt_pk_f16_fp8)
    FPSAN_DEFINE_CVT_PK_F16_FP8(amdgcn_cvt_pk_f16_bf8, fp8_e5m2, __builtin_amdgcn_cvt_pk_f16_bf8)
#undef FPSAN_DEFINE_CVT_PK_F16_FP8
#endif

    // cvt_pk_fp8_f16 / cvt_pk_bf8_f16: pack two f16 to two fp8 bytes in a short
    // (element 0 -> low byte, element 1 -> high byte).
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_pk_fp8_f16)
#define FPSAN_DEFINE_CVT_PK_FP8_F16(name, FP8, BUILTIN)                               \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE short name(Value<v2h_native, S, C> ab)                               \
    {                                                                                 \
        if constexpr(S == Semantics::Native)                                          \
            return BUILTIN(ab.to_float());                                            \
        else                                                                          \
        {                                                                             \
            const auto          a     = fpsan::cast<FP8>(ab.get(0));                  \
            const auto          b     = fpsan::cast<FP8>(ab.get(1));                  \
            const std::uint32_t abyte = static_cast<std::uint8_t>(a.fpsan_payload()); \
            const std::uint32_t bbyte = static_cast<std::uint8_t>(b.fpsan_payload()); \
            return static_cast<short>(abyte | (bbyte << 8));                          \
        }                                                                             \
    }
    FPSAN_DEFINE_CVT_PK_FP8_F16(amdgcn_cvt_pk_fp8_f16, fp8_e4m3, __builtin_amdgcn_cvt_pk_fp8_f16)
    FPSAN_DEFINE_CVT_PK_FP8_F16(amdgcn_cvt_pk_bf8_f16, fp8_e5m2, __builtin_amdgcn_cvt_pk_bf8_f16)
#undef FPSAN_DEFINE_CVT_PK_FP8_F16
#endif

    // cvt_sr_fp8_f16 / cvt_sr_bf8_f16: stochastic-round an f16 to fp8 and splice
    // it into byte ByteIdx of `packed_old`. As with the f32 SR ops, the seed is
    // opaque to the payload ring (the FPSan answer is the deterministic cast).
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_sr_fp8_f16)
#define FPSAN_DEFINE_CVT_SR_FP8_F16(NAME, FP8, BUILTIN)                                            \
    template <int ByteIdx, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE int NAME(Value<_Float16, S, C> val, int packed_old, std::uint32_t seed)           \
    {                                                                                              \
        if constexpr(S == Semantics::Native)                                                       \
            return BUILTIN(val.to_float(), static_cast<int>(seed), packed_old, ByteIdx);           \
        else                                                                                       \
        {                                                                                          \
            (void)seed;                                                                            \
            const auto f8 = fpsan::cast<FP8>(val);                                                 \
            return detail::splice_fp8_byte<ByteIdx>(static_cast<std::uint8_t>(f8.fpsan_payload()), \
                                                    packed_old);                                   \
        }                                                                                          \
    }
    FPSAN_DEFINE_CVT_SR_FP8_F16(amdgcn_cvt_sr_fp8_f16, fp8_e4m3, __builtin_amdgcn_cvt_sr_fp8_f16)
    FPSAN_DEFINE_CVT_SR_FP8_F16(amdgcn_cvt_sr_bf8_f16, fp8_e5m2, __builtin_amdgcn_cvt_sr_bf8_f16)
#undef FPSAN_DEFINE_CVT_SR_FP8_F16
#endif

// cvt_sr_{f16,bf16}_f32: stochastic rounding f32 -> packed f16/bf16, splicing
// into the lo or hi half of a v2 fragment (DstLo selects). gfx950-specific
// ('f32-to-f16bf16-cvt-sr-insts').  Same FPSan story: seed is opaque, the
// payload-domain answer is the deterministic Triton-style cast.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_sr_f16_f32)
    template <bool DstLo, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<v2h_native, S, C> amdgcn_cvt_sr_f16_f32(Value<v2h_native, S, C> old,
                                                               Value<float, S, C>      val,
                                                               std::uint32_t           seed)
    {
        if constexpr(S == Semantics::Native)
        {
            v2h_native r = __builtin_amdgcn_cvt_sr_f16_f32(
                old.to_float(), val.to_float(), seed, DstLo ? false : true);
            return Value<v2h_native, S, C>(r);
        }
        else
        {
            (void)seed;
            auto                    h = fpsan::cast<_Float16>(val);
            Value<v2h_native, S, C> r = old;
            r.set(DstLo ? 0 : 1, h);
            return r;
        }
    }

    template <bool DstLo, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<v2bf_native, S, C> amdgcn_cvt_sr_bf16_f32(Value<v2bf_native, S, C> old,
                                                                 Value<float, S, C>       val,
                                                                 std::uint32_t            seed)
    {
        if constexpr(S == Semantics::Native)
        {
            v2bf_native r = __builtin_amdgcn_cvt_sr_bf16_f32(
                old.to_float(), val.to_float(), seed, DstLo ? false : true);
            return Value<v2bf_native, S, C>(r);
        }
        else
        {
            (void)seed;
            auto                     h = fpsan::cast<__bf16>(val);
            Value<v2bf_native, S, C> r = old;
            r.set(DstLo ? 0 : 1, h);
            return r;
        }
    }
#endif

    // =============================================================================
    // Scaled MX cvt family (cvt_scalef32_*): the gfx950 MX micro-scaled-format
    // conversion intrinsics.
    //
    // These take an additional 'scale' f32 and apply it as part of the conversion
    // -- the OCP MX block-scale model. DIRECTION (verified on gfx950 silicon):
    //   * UNPACK (narrow->f32/f16/bf16) MULTIPLIES by scale: out = val * scale.
    //   * PACK   (f32/f16/bf16->narrow) DIVIDES   by scale: stored = val / scale.
    // so pack(x, s) followed by unpack(s) round-trips. In FPSan mode the scale is
    // a regular Value<float> operand and the mul/div happen in the payload ring.
    //
    // Coverage (all silicon-tested in both modes):
    //   * fp8/bf8: scalar unpack (f32_{fp8,bf8}), scalar pack (pk_{fp8,bf8}_f32),
    //     vector unpack (pk_{f32,f16,bf16}_{fp8,bf8}), vector pack from f16/bf16
    //     (pk_{fp8,bf8}_{f16,bf16}).
    //   * fp4 (e2m1): unpack (pk_{f32,f16,bf16}_fp4), pack (pk_fp4_{f32,f16,bf16}),
    //     SR pack (sr_pk_fp4_f32).
    //   * fp6 (e2m3) / bf6 (e3m2): unpack (pk32_{f32,f16,bf16}_{fp6,bf6}), pack
    //     (2xpk16_{fp6,bf6}_f32 interleaved + pk32_{fp6,bf6}_{f16,bf16}
    //     contiguous), SR pack (sr_pk32_{fp6,bf6}_{f32,f16,bf16}).
    //   * tied scalar unpack to f16: f16_{fp8,bf8} (byte-select, lo/hi half).
    //   * scaled SR pack to a byte: sr_{fp8,bf8}_{f32,f16,bf16}.
    //   * scaled SR pack to fp4 nibble pair: sr_pk_fp4_{f32,f16,bf16}.
    // This covers every gfx950-selectable cvt_scalef32 op (see the file header
    // for the gfx1250-only forms that are intentionally out of scope).
    // =============================================================================

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_f32_fp8)
    // Scaled scalar unpack: byte ByteIdx of `packed`, treated as fp8 e4m3,
    // converted to f32 with a 2^scale multiplier.
    template <int ByteIdx, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<float, S, C> amdgcn_cvt_scalef32_f32_fp8(int                packed,
                                                                Value<float, S, C> scale)
    {
        if constexpr(S == Semantics::Native)
            return Value<float, S, C>(
                __builtin_amdgcn_cvt_scalef32_f32_fp8(packed, scale.to_float(), ByteIdx));
        else
        {
            auto v = detail::unpack_byte_to_fp8<ByteIdx, fp8_e4m3, S, C>(packed);
            return fpsan::cast<float>(v) * scale;
        }
    }

    template <int ByteIdx, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<float, S, C> amdgcn_cvt_scalef32_f32_bf8(int                packed,
                                                                Value<float, S, C> scale)
    {
        if constexpr(S == Semantics::Native)
            return Value<float, S, C>(
                __builtin_amdgcn_cvt_scalef32_f32_bf8(packed, scale.to_float(), ByteIdx));
        else
        {
            auto v = detail::unpack_byte_to_fp8<ByteIdx, fp8_e5m2, S, C>(packed);
            return fpsan::cast<float>(v) * scale;
        }
    }
#endif

// ---- Scaled scalar unpack to f16 (tied dst): byte ByteIdx of `src`, decoded
// as fp8/bf8, * scale, written into the lo (Hi=false) or hi (Hi=true) half of a
// v2f16; the OTHER half is preserved from `old`. Silicon-verified on gfx950:
// cvt_scalef32_f16_fp8(old, src, scale, ByteIdx, Hi) reads byte ByteIdx (same
// decode as cvt_f32_fp8), MULTIPLIES by scale (UNPACK direction), and Hi
// selects the destination half (false->lane0, true->lane1). The sibling vector
// unpack cvt_scalef32_pk_f16_fp8 does a whole 16-bit word; this one does one
// byte.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_f16_fp8)
#define FPSAN_DEFINE_CVT_SCALEF32_F16_FP8(NAME, FP8, BUILTIN)                                \
    template <int         ByteIdx,                                                           \
              bool        Hi,                                                                \
              Semantics   S = Semantics::Native,                                             \
              Conversions C = Conversions::Explicit>                                         \
    FPSAN_DEVICE Value<v2h_native, S, C> NAME(                                               \
        Value<v2h_native, S, C> old, int src, Value<float, S, C> scale)                      \
    {                                                                                        \
        if constexpr(S == Semantics::Native)                                                 \
        {                                                                                    \
            v2h_native r = BUILTIN(old.to_float(), src, scale.to_float(), ByteIdx, Hi);      \
            return Value<v2h_native, S, C>(r);                                               \
        }                                                                                    \
        else                                                                                 \
        {                                                                                    \
            Value<v2h_native, S, C> r = old;                                                 \
            const auto              v = detail::unpack_byte_to_fp8<ByteIdx, FP8, S, C>(src); \
            r.set(Hi ? 1 : 0, fpsan::cast<_Float16>(fpsan::cast<float>(v) * scale));         \
            return r;                                                                        \
        }                                                                                    \
    }
    FPSAN_DEFINE_CVT_SCALEF32_F16_FP8(amdgcn_cvt_scalef32_f16_fp8,
                                      fp8_e4m3,
                                      __builtin_amdgcn_cvt_scalef32_f16_fp8)
    FPSAN_DEFINE_CVT_SCALEF32_F16_FP8(amdgcn_cvt_scalef32_f16_bf8,
                                      fp8_e5m2,
                                      __builtin_amdgcn_cvt_scalef32_f16_bf8)
#undef FPSAN_DEFINE_CVT_SCALEF32_F16_FP8
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_fp8_f32)
    // Scaled packed pack: two f32s and a scale, spliced into the lo or hi half
    // of a packed int (cvt_pk_fp8_f32's scaled cousin).
    template <bool DstLo, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE int amdgcn_cvt_scalef32_pk_fp8_f32(Value<float, S, C> a,
                                                    Value<float, S, C> b,
                                                    Value<float, S, C> scale,
                                                    int                old)
    {
        if constexpr(S == Semantics::Native)
        {
            // Builtin takes/returns _Vector<2,short>; bit-identical to our int. The
            // word-select arg is false=low, true=high (silicon-verified), so write the
            // low half (DstLo=true) with !DstLo -- matching pack_fp8_pair below.
            const auto oldv = __builtin_bit_cast(v2s16_native, old);
            const auto r    = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
                oldv, a.to_float(), b.to_float(), scale.to_float(), !DstLo);
            return __builtin_bit_cast(int, r);
        }
        else
        {
            // MX block-scale: PACK stores value/scale (verified on gfx950: pack 4.0
            // with scale=2.0 yields fp8 code for 2.0). The matching UNPACK multiplies
            // by scale, so pack(x,s) then unpack(s) round-trips. (The unpack wrappers
            // multiply; only this pack direction divides.)
            auto       sa = a / scale;
            auto       sb = b / scale;
            const auto af = fpsan::cast<fp8_e4m3>(sa);
            const auto bf = fpsan::cast<fp8_e4m3>(sb);
            return detail::pack_fp8_pair<DstLo>(static_cast<std::uint8_t>(af.fpsan_payload()),
                                                static_cast<std::uint8_t>(bf.fpsan_payload()),
                                                old);
        }
    }

    template <bool DstLo, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE int amdgcn_cvt_scalef32_pk_bf8_f32(Value<float, S, C> a,
                                                    Value<float, S, C> b,
                                                    Value<float, S, C> scale,
                                                    int                old)
    {
        if constexpr(S == Semantics::Native)
        {
            const auto oldv = __builtin_bit_cast(v2s16_native, old);
            const auto r    = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(
                oldv, a.to_float(), b.to_float(), scale.to_float(), !DstLo);
            return __builtin_bit_cast(int, r);
        }
        else
        {
            // PACK stores value/scale (see amdgcn_cvt_scalef32_pk_fp8_f32).
            auto       sa = a / scale;
            auto       sb = b / scale;
            const auto af = fpsan::cast<fp8_e5m2>(sa);
            const auto bf = fpsan::cast<fp8_e5m2>(sb);
            return detail::pack_fp8_pair<DstLo>(static_cast<std::uint8_t>(af.fpsan_payload()),
                                                static_cast<std::uint8_t>(bf.fpsan_payload()),
                                                old);
        }
    }
#endif

// ---- Scaled VECTOR unpack: word WordSel of `packed` (2 fp8/bf8 bytes) -> a
// pair of f32/f16/bf16, each * scale. The scalar cvt_scalef32_f32_{fp8,bf8}
// above does one byte; these do a 16-bit word (bytes 2*WordSel, 2*WordSel+1),
// mirroring the non-scaled cvt_pk_f32_{fp8,bf8} but with the 2^scale multiply.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_f32_fp8)
#define FPSAN_DEFINE_CVT_SCALEF32_UNPACK_FP8(NAME, FP8, DstFT, VEC, BUILTIN)                      \
    template <bool        WordSel,                                                                \
              Semantics   S = Semantics::Native,                                                  \
              Conversions C = Conversions::Explicit>                                              \
    FPSAN_DEVICE Value<VEC, S, C> NAME(int packed, Value<float, S, C> scale)                      \
    {                                                                                             \
        if constexpr(S == Semantics::Native)                                                      \
        {                                                                                         \
            return Value<VEC, S, C>(BUILTIN(packed, scale.to_float(), WordSel));                  \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            Value<VEC, S, C> r{};                                                                 \
            const auto       v0 = detail::unpack_byte_to_fp8<2 * WordSel, FP8, S, C>(packed);     \
            const auto       v1 = detail::unpack_byte_to_fp8<2 * WordSel + 1, FP8, S, C>(packed); \
            r.set(0, fpsan::cast<DstFT>(fpsan::cast<float>(v0) * scale));                         \
            r.set(1, fpsan::cast<DstFT>(fpsan::cast<float>(v1) * scale));                         \
            return r;                                                                             \
        }                                                                                         \
    }
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_FP8(amdgcn_cvt_scalef32_pk_f32_fp8,
                                         fp8_e4m3,
                                         float,
                                         v2f_native,
                                         __builtin_amdgcn_cvt_scalef32_pk_f32_fp8)
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_FP8(amdgcn_cvt_scalef32_pk_f32_bf8,
                                         fp8_e5m2,
                                         float,
                                         v2f_native,
                                         __builtin_amdgcn_cvt_scalef32_pk_f32_bf8)
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_FP8(amdgcn_cvt_scalef32_pk_f16_fp8,
                                         fp8_e4m3,
                                         _Float16,
                                         v2h_native,
                                         __builtin_amdgcn_cvt_scalef32_pk_f16_fp8)
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_FP8(amdgcn_cvt_scalef32_pk_f16_bf8,
                                         fp8_e5m2,
                                         _Float16,
                                         v2h_native,
                                         __builtin_amdgcn_cvt_scalef32_pk_f16_bf8)
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_FP8(amdgcn_cvt_scalef32_pk_bf16_fp8,
                                         fp8_e4m3,
                                         __bf16,
                                         v2bf_native,
                                         __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8)
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_FP8(amdgcn_cvt_scalef32_pk_bf16_bf8,
                                         fp8_e5m2,
                                         __bf16,
                                         v2bf_native,
                                         __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8)
#undef FPSAN_DEFINE_CVT_SCALEF32_UNPACK_FP8

// ---- Scaled VECTOR pack from f16/bf16: 2 src values -> word DstLo of a packed
// int (2 fp8/bf8 bytes), each value / scale. cvt_scalef32_pk_{fp8,bf8}_{f16,
// bf16}. Same divide-by-scale PACK direction as the f32 scalar pack; the
// builtin's old/result are _Vector<2,short>, bit-identical to our int.
#define FPSAN_DEFINE_CVT_SCALEF32_PACK_FP8(NAME, FP8, SrcVEC, BUILTIN)                            \
    template <bool DstLo, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE int NAME(int old, Value<SrcVEC, S, C> v, Value<float, S, C> scale)               \
    {                                                                                             \
        if constexpr(S == Semantics::Native)                                                      \
        {                                                                                         \
            const auto oldv = __builtin_bit_cast(v2s16_native, old);                              \
            const auto r    = BUILTIN(oldv, v.to_float(), scale.to_float(), !DstLo);              \
            return __builtin_bit_cast(int, r);                                                    \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            const auto a  = fpsan::cast<float>(v.get(0)) / scale;                                 \
            const auto b  = fpsan::cast<float>(v.get(1)) / scale;                                 \
            const auto af = fpsan::cast<FP8>(a);                                                  \
            const auto bf = fpsan::cast<FP8>(b);                                                  \
            return detail::pack_fp8_pair<DstLo>(static_cast<std::uint8_t>(af.fpsan_payload()),    \
                                                static_cast<std::uint8_t>(bf.fpsan_payload()),    \
                                                old);                                             \
        }                                                                                         \
    }
    FPSAN_DEFINE_CVT_SCALEF32_PACK_FP8(amdgcn_cvt_scalef32_pk_fp8_f16,
                                       fp8_e4m3,
                                       v2h_native,
                                       __builtin_amdgcn_cvt_scalef32_pk_fp8_f16)
    FPSAN_DEFINE_CVT_SCALEF32_PACK_FP8(amdgcn_cvt_scalef32_pk_fp8_bf16,
                                       fp8_e4m3,
                                       v2bf_native,
                                       __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16)
    FPSAN_DEFINE_CVT_SCALEF32_PACK_FP8(amdgcn_cvt_scalef32_pk_bf8_f16,
                                       fp8_e5m2,
                                       v2h_native,
                                       __builtin_amdgcn_cvt_scalef32_pk_bf8_f16)
    FPSAN_DEFINE_CVT_SCALEF32_PACK_FP8(amdgcn_cvt_scalef32_pk_bf8_bf16,
                                       fp8_e5m2,
                                       v2bf_native,
                                       __builtin_amdgcn_cvt_scalef32_pk_bf8_bf16)
#undef FPSAN_DEFINE_CVT_SCALEF32_PACK_FP8
#endif // __has_builtin(cvt_scalef32_pk_f32_fp8)

// =============================================================================
// Scaled stochastic-rounding pack to fp8/bf8 (cvt_scalef32_sr_{fp8,bf8}_{f32,
// f16,bf16}; gfx950). Convert one scalar `val` (f32/f16/bf16) to fp8/bf8 with a
// 2^scale divide (PACK direction), stochastic-round, and splice the byte into
// position ByteIdx of `old`. Silicon-verified on gfx950: byte ByteIdx is
// written directly (NO +1 offset, unlike the non-scaled cvt_sr_fp8) and the
// stored value is val/scale (e.g. sr(4.0, scale=2)=fp8 code of 2.0). FPSan
// model: seed is opaque to the deterministic Triton truncate, so the answer is
// the same byte the scaled non-SR pack would write -- cast<FP8>(val/scale).
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_sr_fp8_f32)
#define FPSAN_DEFINE_CVT_SCALEF32_SR_FP8(NAME, FP8, SRC, BUILTIN)                                  \
    template <int ByteIdx, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE int NAME(                                                                         \
        int old, Value<SRC, S, C> val, std::uint32_t seed, Value<float, S, C> scale)               \
    {                                                                                              \
        if constexpr(S == Semantics::Native)                                                       \
            return BUILTIN(old, val.to_float(), seed, scale.to_float(), ByteIdx);                  \
        else                                                                                       \
        {                                                                                          \
            (void)seed;                                                                            \
            const auto f8 = fpsan::cast<FP8>(fpsan::cast<float>(val) / scale);                     \
            return detail::splice_fp8_byte<ByteIdx>(static_cast<std::uint8_t>(f8.fpsan_payload()), \
                                                    old);                                          \
        }                                                                                          \
    }
    FPSAN_DEFINE_CVT_SCALEF32_SR_FP8(amdgcn_cvt_scalef32_sr_fp8_f32,
                                     fp8_e4m3,
                                     float,
                                     __builtin_amdgcn_cvt_scalef32_sr_fp8_f32)
    FPSAN_DEFINE_CVT_SCALEF32_SR_FP8(amdgcn_cvt_scalef32_sr_fp8_f16,
                                     fp8_e4m3,
                                     _Float16,
                                     __builtin_amdgcn_cvt_scalef32_sr_fp8_f16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_FP8(amdgcn_cvt_scalef32_sr_fp8_bf16,
                                     fp8_e4m3,
                                     __bf16,
                                     __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_FP8(amdgcn_cvt_scalef32_sr_bf8_f32,
                                     fp8_e5m2,
                                     float,
                                     __builtin_amdgcn_cvt_scalef32_sr_bf8_f32)
    FPSAN_DEFINE_CVT_SCALEF32_SR_FP8(amdgcn_cvt_scalef32_sr_bf8_f16,
                                     fp8_e5m2,
                                     _Float16,
                                     __builtin_amdgcn_cvt_scalef32_sr_bf8_f16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_FP8(amdgcn_cvt_scalef32_sr_bf8_bf16,
                                     fp8_e5m2,
                                     __bf16,
                                     __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16)
#undef FPSAN_DEFINE_CVT_SCALEF32_SR_FP8
#endif

    // =============================================================================
    // Scaled MX fp4 (e2m1) conversions (fp4-cvt-scale-insts; gfx950).
    //
    // fp4 packs 8 nibbles into a u32; the _Constant `Sel` (0..3) names a nibble
    // PAIR == byte `Sel` (verified on silicon): a -> low nibble (2*Sel), b -> high
    // nibble (2*Sel+1). Scale DIRECTION matches the fp8 family: UNPACK multiplies
    // by scale, PACK divides by scale.
    //
    // In FPSan mode the packed u32 holds fp4 PAYLOADS (4-bit) per nibble, exactly
    // as the fp8 cvt wrappers treat bytes as fp8 payloads. The cast between an fp4
    // nibble and f32 is the Triton sign-resize on the payload (sign-extend 4->32 on
    // widen; keep the low 4 bits on narrow) -- there is no Value<fp4> (sub-byte is
    // never a Value element type), so we resize the payload directly. All scale
    // arithmetic happens in the f32 payload ring (the scale operand is f32), with
    // the f16/bf16 in/out variants widening to / narrowing from f32 around it.
    // =============================================================================
    namespace detail
    {

        // Sign-resize a 4-bit fp4 payload (low nibble of `nib`) to a 32-bit f32 payload
        // (FPSan widen, == fpsan::cast<float> on a hypothetical Value<fp4>).
        template <Semantics S, Conversions C>
        FPSAN_DEVICE Value<float, S, C> fp4_nibble_to_f32(std::uint32_t nib)
        {
            return subbyte_widen<4, S, C>(nib);
        }
        // Narrow a 32-bit f32 payload to a 4-bit fp4 payload (FPSan narrow == keep low
        // 4 bits of the signed payload).
        template <Semantics S, Conversions C>
        FPSAN_DEVICE std::uint32_t f32_to_fp4_nibble(Value<float, S, C> v)
        {
            return static_cast<std::uint32_t>(v.fpsan_payload()) & 0xFu;
        }
        // Splice a 4-bit value into nibble pair `Sel` (low nibble = a, high = b) of an
        // existing u32, preserving the other six nibbles.
        template <int Sel>
        FPSAN_DEVICE std::uint32_t
                     splice_fp4_pair(std::uint32_t a4, std::uint32_t b4, std::uint32_t old)
        {
            static_assert(Sel >= 0 && Sel < 4, "fp4 Sel must be 0..3");
            const std::uint32_t pair = (a4 & 0xFu) | ((b4 & 0xFu) << 4);
            const std::uint32_t mask = ~(std::uint32_t{0xFFu} << (8 * Sel));
            return (old & mask) | (pair << (8 * Sel));
        }
    } // namespace detail

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4)
// ---- Unpack: fp4 nibble pair `Sel` of `packed` -> 2 dst values * scale. -----
#define FPSAN_DEFINE_CVT_SCALEF32_UNPACK_FP4(NAME, DstFT, VEC, BUILTIN)                        \
    template <int Sel, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE Value<VEC, S, C> NAME(std::uint32_t packed, Value<float, S, C> scale)         \
    {                                                                                          \
        if constexpr(S == Semantics::Native)                                                   \
        {                                                                                      \
            VEC r = BUILTIN(packed, scale.to_float(), Sel);                                    \
            return Value<VEC, S, C>(r);                                                        \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
            const std::uint32_t n0 = (packed >> (8 * Sel)) & 0xFu;                             \
            const std::uint32_t n1 = (packed >> (8 * Sel + 4)) & 0xFu;                         \
            Value<VEC, S, C>    r{};                                                           \
            r.set(0, fpsan::cast<DstFT>(detail::fp4_nibble_to_f32<S, C>(n0) * scale));         \
            r.set(1, fpsan::cast<DstFT>(detail::fp4_nibble_to_f32<S, C>(n1) * scale));         \
            return r;                                                                          \
        }                                                                                      \
    }
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_FP4(amdgcn_cvt_scalef32_pk_f32_fp4,
                                         float,
                                         v2f_native,
                                         __builtin_amdgcn_cvt_scalef32_pk_f32_fp4)
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_FP4(amdgcn_cvt_scalef32_pk_f16_fp4,
                                         _Float16,
                                         v2h_native,
                                         __builtin_amdgcn_cvt_scalef32_pk_f16_fp4)
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_FP4(amdgcn_cvt_scalef32_pk_bf16_fp4,
                                         __bf16,
                                         v2bf_native,
                                         __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4)
#undef FPSAN_DEFINE_CVT_SCALEF32_UNPACK_FP4

    // ---- Pack: 2 f32 -> fp4 nibble pair `Sel` of `old`, each value / scale. -----
    template <int Sel, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE std::uint32_t amdgcn_cvt_scalef32_pk_fp4_f32(std::uint32_t      old,
                                                              Value<float, S, C> a,
                                                              Value<float, S, C> b,
                                                              Value<float, S, C> scale)
    {
        if constexpr(S == Semantics::Native)
            return __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                old, a.to_float(), b.to_float(), scale.to_float(), Sel);
        else
            return detail::splice_fp4_pair<Sel>(detail::f32_to_fp4_nibble<S, C>(a / scale),
                                                detail::f32_to_fp4_nibble<S, C>(b / scale),
                                                old);
    }

// f16/bf16 pack: widen the v2 input to f32, divide by scale, narrow to fp4.
#define FPSAN_DEFINE_CVT_SCALEF32_PACK_FP4(NAME, SrcFT, VEC, BUILTIN)                          \
    template <int Sel, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE std::uint32_t NAME(                                                           \
        std::uint32_t old, Value<VEC, S, C> v, Value<float, S, C> scale)                       \
    {                                                                                          \
        if constexpr(S == Semantics::Native)                                                   \
        {                                                                                      \
            return BUILTIN(old, v.to_float(), scale.to_float(), Sel);                          \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
            const auto a = fpsan::cast<float>(v.get(0)) / scale;                               \
            const auto b = fpsan::cast<float>(v.get(1)) / scale;                               \
            return detail::splice_fp4_pair<Sel>(                                               \
                detail::f32_to_fp4_nibble<S, C>(a), detail::f32_to_fp4_nibble<S, C>(b), old);  \
        }                                                                                      \
    }
    FPSAN_DEFINE_CVT_SCALEF32_PACK_FP4(amdgcn_cvt_scalef32_pk_fp4_f16,
                                       _Float16,
                                       v2h_native,
                                       __builtin_amdgcn_cvt_scalef32_pk_fp4_f16)
    FPSAN_DEFINE_CVT_SCALEF32_PACK_FP4(amdgcn_cvt_scalef32_pk_fp4_bf16,
                                       __bf16,
                                       v2bf_native,
                                       __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16)
#undef FPSAN_DEFINE_CVT_SCALEF32_PACK_FP4

// ---- Stochastic-rounding fp4 pack (sr_pk_fp4_*): seed is opaque to the
// payload-domain answer (deterministic Triton truncate), exactly like the
// fp8 SR family. The PR flags the reversed-operand encoding (data in src1);
// the wrapper passes operands in the builtin's order so the Float path is the
// hardware truth and the FPSan path is the matching deterministic pack.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32)
    template <int Sel, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE std::uint32_t amdgcn_cvt_scalef32_sr_pk_fp4_f32(std::uint32_t      old,
                                                                 Value<float, S, C> a,
                                                                 Value<float, S, C> b,
                                                                 std::uint32_t      seed,
                                                                 Value<float, S, C> scale)
    {
        if constexpr(S == Semantics::Native)
        {
            v2f_native v{a.to_float(), b.to_float()};
            return __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(old, v, seed, scale.to_float(), Sel);
        }
        else
        {
            (void)seed;
            return detail::splice_fp4_pair<Sel>(detail::f32_to_fp4_nibble<S, C>(a / scale),
                                                detail::f32_to_fp4_nibble<S, C>(b / scale),
                                                old);
        }
    }
#endif

// ---- SR pack to fp4 from f16/bf16 (sr_pk_fp4_{f16,bf16}): widen the v2 source
// to f32, divide by scale, narrow to fp4 nibble pair Sel (a -> low nibble
// 2*Sel, b -> high nibble 2*Sel+1; silicon-verified, no operand swap). seed is
// opaque to the deterministic Triton truncate (same as the f32-source SR pack).
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16)
#define FPSAN_DEFINE_CVT_SCALEF32_SR_PK_FP4(NAME, VEC, BUILTIN)                                \
    template <int Sel, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE std::uint32_t NAME(                                                           \
        std::uint32_t old, Value<VEC, S, C> v, std::uint32_t seed, Value<float, S, C> scale)   \
    {                                                                                          \
        if constexpr(S == Semantics::Native)                                                   \
        {                                                                                      \
            return BUILTIN(old, v.to_float(), seed, scale.to_float(), Sel);                    \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
            (void)seed;                                                                        \
            return detail::splice_fp4_pair<Sel>(                                               \
                detail::f32_to_fp4_nibble<S, C>(fpsan::cast<float>(v.get(0)) / scale),         \
                detail::f32_to_fp4_nibble<S, C>(fpsan::cast<float>(v.get(1)) / scale),         \
                old);                                                                          \
        }                                                                                      \
    }
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK_FP4(amdgcn_cvt_scalef32_sr_pk_fp4_f16,
                                        v2h_native,
                                        __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK_FP4(amdgcn_cvt_scalef32_sr_pk_fp4_bf16,
                                        v2bf_native,
                                        __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_bf16)
#undef FPSAN_DEFINE_CVT_SCALEF32_SR_PK_FP4
#endif
#endif // __has_builtin(cvt_scalef32_pk_f32_fp4)

    // =============================================================================
    // Scaled MX fp6 (e2m3) / bf6 (e3m2) conversions (gfx950).
    //
    // 32 six-bit codes pack into a v6u32 (6 DWORDs = 192 bits). Layouts verified on
    // silicon (see cvt_scalef32_mx_gfx950_test.cpp):
    //   * pk32_*       : CONTIGUOUS little-endian 6-bit stream, field i = elem i.
    //   * 2xpk16_*_f32 : two v16 groups INTERLEAVED -- field 2k = lo[k],
    //   2k+1=hi[k].
    //   * sr_pk32_*    : contiguous (same as pk32); seed is opaque to exact inputs.
    // Scale direction matches the rest of the family: UNPACK multiplies, PACK
    // divides. In FPSan mode the v6u32 holds 6-bit fp6/bf6 PAYLOADS; widen sign-
    // extends 6->32, narrow keeps the low 6 bits, all in the f32 payload ring.
    // =============================================================================
    using v32f_native     = float __attribute__((ext_vector_type(32)));
    using v32h_native     = _Float16 __attribute__((ext_vector_type(32)));
    using v32bf_native    = __bf16 __attribute__((ext_vector_type(32)));
    using v16f_native_cvt = float __attribute__((ext_vector_type(16)));
    using v6u32_native    = unsigned __attribute__((ext_vector_type(6)));

    namespace detail
    {
        // Contiguous little-endian 6-bit field i of a packed u32 stream. Templated
        // over the packed vector type so the same routine serves the 32-code v6u32
        // (192-bit) stream and the gfx1250 16-code v3u32 (96-bit) stream; both keep
        // every access in-bounds (code 15 ends at bit 95 in word 2, code 31 at bit
        // 191 in word 5, neither straddles past its last word).
        template <class V>
        FPSAN_DEVICE inline std::uint32_t extract6(const V& w, int i)
        {
            const int     p = i * 6, wi = p >> 5, off = p & 31;
            std::uint32_t v = w[wi] >> off;
            if(off > 26)
                v |= w[wi + 1] << (32 - off);
            return v & 0x3Fu;
        }
        template <class V>
        FPSAN_DEVICE inline void insert6(V& w, int i, std::uint32_t v6)
        {
            v6 &= 0x3Fu;
            const int p = i * 6, wi = p >> 5, off = p & 31;
            w[wi] |= v6 << off;
            if(off > 26)
                w[wi + 1] |= v6 >> (32 - off);
        }
        // Sign-resize a 6-bit payload to a 32-bit f32 payload (FPSan widen).
        template <Semantics S, Conversions C>
        FPSAN_DEVICE Value<float, S, C> sub6_to_f32(std::uint32_t f)
        {
            return subbyte_widen<6, S, C>(f);
        }
        template <Semantics S, Conversions C>
        FPSAN_DEVICE std::uint32_t f32_to_sub6(Value<float, S, C> v)
        {
            return static_cast<std::uint32_t>(v.fpsan_payload()) & 0x3Fu;
        }
    } // namespace detail

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_pk32_f32_fp6)
// ---- Unpack v6u32 (32 fp6/bf6 codes) -> v32 dst, each * scale (contiguous).
#define FPSAN_DEFINE_CVT_SCALEF32_UNPACK_PK32(NAME, DstFT, VEC, BUILTIN)                        \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>           \
    FPSAN_DEVICE Value<VEC, S, C> NAME(v6u32_native packed, Value<float, S, C> scale)           \
    {                                                                                           \
        if constexpr(S == Semantics::Native)                                                    \
        {                                                                                       \
            VEC r = BUILTIN(packed, scale.to_float());                                          \
            return Value<VEC, S, C>(r);                                                         \
        }                                                                                       \
        else                                                                                    \
        {                                                                                       \
            Value<VEC, S, C> r{};                                                               \
            for(int i = 0; i < 32; ++i)                                                         \
                r.set(i,                                                                        \
                      fpsan::cast<DstFT>(detail::sub6_to_f32<S, C>(detail::extract6(packed, i)) \
                                         * scale));                                             \
            return r;                                                                           \
        }                                                                                       \
    }
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_PK32(amdgcn_cvt_scalef32_pk32_f32_fp6,
                                          float,
                                          v32f_native,
                                          __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6)
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_PK32(amdgcn_cvt_scalef32_pk32_f32_bf6,
                                          float,
                                          v32f_native,
                                          __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6)
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_PK32(amdgcn_cvt_scalef32_pk32_f16_fp6,
                                          _Float16,
                                          v32h_native,
                                          __builtin_amdgcn_cvt_scalef32_pk32_f16_fp6)
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_PK32(amdgcn_cvt_scalef32_pk32_f16_bf6,
                                          _Float16,
                                          v32h_native,
                                          __builtin_amdgcn_cvt_scalef32_pk32_f16_bf6)
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_PK32(amdgcn_cvt_scalef32_pk32_bf16_fp6,
                                          __bf16,
                                          v32bf_native,
                                          __builtin_amdgcn_cvt_scalef32_pk32_bf16_fp6)
    FPSAN_DEFINE_CVT_SCALEF32_UNPACK_PK32(amdgcn_cvt_scalef32_pk32_bf16_bf6,
                                          __bf16,
                                          v32bf_native,
                                          __builtin_amdgcn_cvt_scalef32_pk32_bf16_bf6)
#undef FPSAN_DEFINE_CVT_SCALEF32_UNPACK_PK32

// ---- Pack two v16 f32 groups -> v6u32 (32 fp6/bf6), interleaved, each /scale.
#define FPSAN_DEFINE_CVT_SCALEF32_2XPK16(NAME, BUILTIN)                                        \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>          \
    FPSAN_DEVICE v6u32_native NAME(Value<v16f_native_cvt, S, C> lo,                            \
                                   Value<v16f_native_cvt, S, C> hi,                            \
                                   Value<float, S, C>           scale)                         \
    {                                                                                          \
        if constexpr(S == Semantics::Native)                                                   \
        {                                                                                      \
            return BUILTIN(lo.to_float(), hi.to_float(), scale.to_float());                    \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
            v6u32_native out{};                                                                \
            for(int k = 0; k < 16; ++k)                                                        \
            {                                                                                  \
                detail::insert6(out, 2 * k, detail::f32_to_sub6<S, C>(lo.get(k) / scale));     \
                detail::insert6(out, 2 * k + 1, detail::f32_to_sub6<S, C>(hi.get(k) / scale)); \
            }                                                                                  \
            return out;                                                                        \
        }                                                                                      \
    }
    FPSAN_DEFINE_CVT_SCALEF32_2XPK16(amdgcn_cvt_scalef32_2xpk16_fp6_f32,
                                     __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32)
    FPSAN_DEFINE_CVT_SCALEF32_2XPK16(amdgcn_cvt_scalef32_2xpk16_bf6_f32,
                                     __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32)
#undef FPSAN_DEFINE_CVT_SCALEF32_2XPK16

// ---- Pack v32 f16/bf16 -> v6u32 (contiguous), each /scale.
#define FPSAN_DEFINE_CVT_SCALEF32_PK32_PACK(NAME, SrcFT, VEC, BUILTIN)                        \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>         \
    FPSAN_DEVICE v6u32_native NAME(Value<VEC, S, C> v, Value<float, S, C> scale)              \
    {                                                                                         \
        if constexpr(S == Semantics::Native)                                                  \
        {                                                                                     \
            return BUILTIN(v.to_float(), scale.to_float());                                   \
        }                                                                                     \
        else                                                                                  \
        {                                                                                     \
            v6u32_native out{};                                                               \
            for(int i = 0; i < 32; ++i)                                                       \
                detail::insert6(                                                              \
                    out, i, detail::f32_to_sub6<S, C>(fpsan::cast<float>(v.get(i)) / scale)); \
            return out;                                                                       \
        }                                                                                     \
    }
    FPSAN_DEFINE_CVT_SCALEF32_PK32_PACK(amdgcn_cvt_scalef32_pk32_fp6_f16,
                                        _Float16,
                                        v32h_native,
                                        __builtin_amdgcn_cvt_scalef32_pk32_fp6_f16)
    FPSAN_DEFINE_CVT_SCALEF32_PK32_PACK(amdgcn_cvt_scalef32_pk32_bf6_f16,
                                        _Float16,
                                        v32h_native,
                                        __builtin_amdgcn_cvt_scalef32_pk32_bf6_f16)
    FPSAN_DEFINE_CVT_SCALEF32_PK32_PACK(amdgcn_cvt_scalef32_pk32_fp6_bf16,
                                        __bf16,
                                        v32bf_native,
                                        __builtin_amdgcn_cvt_scalef32_pk32_fp6_bf16)
    FPSAN_DEFINE_CVT_SCALEF32_PK32_PACK(amdgcn_cvt_scalef32_pk32_bf6_bf16,
                                        __bf16,
                                        v32bf_native,
                                        __builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16)
#undef FPSAN_DEFINE_CVT_SCALEF32_PK32_PACK

// ---- Stochastic-rounding contiguous pack v32 f32 -> v6u32 (seed opaque).
#define FPSAN_DEFINE_CVT_SCALEF32_SR_PK32(NAME, BUILTIN)                              \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE v6u32_native NAME(                                                   \
        Value<v32f_native, S, C> v, std::uint32_t seed, Value<float, S, C> scale)     \
    {                                                                                 \
        if constexpr(S == Semantics::Native)                                          \
        {                                                                             \
            return BUILTIN(v.to_float(), seed, scale.to_float());                     \
        }                                                                             \
        else                                                                          \
        {                                                                             \
            (void)seed;                                                               \
            v6u32_native out{};                                                       \
            for(int i = 0; i < 32; ++i)                                               \
                detail::insert6(out, i, detail::f32_to_sub6<S, C>(v.get(i) / scale)); \
            return out;                                                               \
        }                                                                             \
    }
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f32)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK32(amdgcn_cvt_scalef32_sr_pk32_fp6_f32,
                                      __builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f32)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK32(amdgcn_cvt_scalef32_sr_pk32_bf6_f32,
                                      __builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f32)
#endif

// ---- SR contiguous pack v32 f16/bf16 -> v6u32 (sr_pk32_{fp6,bf6}_{f16,bf16});
// widen each source lane to f32, divide by scale, narrow to 6 bits, contiguous
// insert. Silicon-verified contiguous (bit-identical to non-SR pk32) and seed
// opaque to the deterministic point.
#define FPSAN_DEFINE_CVT_SCALEF32_SR_PK32_SRC(NAME, VEC, BUILTIN)                             \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>         \
    FPSAN_DEVICE v6u32_native NAME(                                                           \
        Value<VEC, S, C> v, std::uint32_t seed, Value<float, S, C> scale)                     \
    {                                                                                         \
        if constexpr(S == Semantics::Native)                                                  \
        {                                                                                     \
            return BUILTIN(v.to_float(), seed, scale.to_float());                             \
        }                                                                                     \
        else                                                                                  \
        {                                                                                     \
            (void)seed;                                                                       \
            v6u32_native out{};                                                               \
            for(int i = 0; i < 32; ++i)                                                       \
                detail::insert6(                                                              \
                    out, i, detail::f32_to_sub6<S, C>(fpsan::cast<float>(v.get(i)) / scale)); \
            return out;                                                                       \
        }                                                                                     \
    }
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK32_SRC(amdgcn_cvt_scalef32_sr_pk32_fp6_f16,
                                          v32h_native,
                                          __builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK32_SRC(amdgcn_cvt_scalef32_sr_pk32_fp6_bf16,
                                          v32bf_native,
                                          __builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_bf16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK32_SRC(amdgcn_cvt_scalef32_sr_pk32_bf6_f16,
                                          v32h_native,
                                          __builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK32_SRC(amdgcn_cvt_scalef32_sr_pk32_bf6_bf16,
                                          v32bf_native,
                                          __builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_bf16)
#endif
#undef FPSAN_DEFINE_CVT_SCALEF32_SR_PK32_SRC
#undef FPSAN_DEFINE_CVT_SCALEF32_SR_PK32
#endif // __has_builtin(cvt_scalef32_pk32_f32_fp6)

    // =============================================================================
    // gfx1250 WIDE block-scale PACK family (cvt_scalef32_pk8 / pk16 [+ sr];
    // "gfx1250-insts").
    //
    // gfx1250 adds fixed-width 8-lane (pk8) and 16-lane (pk16) block-scale PACK
    // ops: a whole bundle of f32/f16/bf16 source lanes plus a single f32 block
    // scale, packed into the narrow output register. Same intrinsic family
    // (cvt_scalef32_*) and same f32 scale operand as the gfx950 pack ops, so the
    // DIRECTION is the (gfx950-silicon-verified) MX pack convention: PACK DIVIDES
    // by scale -- stored = encode(val / scale). With scale == 1 this is a pure OCP
    // encode, which the golden tests assert lane-by-lane against the host
    // encoders (f32_to_narrow), independent of the device builtin, so any
    // implementation that diverges from them shows up as a failure.
    //
    // Packed output layout is the contiguous little-endian narrow stream:
    //   * pk8  fp8/bf8 : 8 bytes   -> v2u32  (byte i   = lane i).
    //   * pk8  fp4     : 8 nibbles -> u32    (nibble i = lane i).
    //   * pk16 fp6/bf6 : 16 codes  -> v3u32  (6-bit field i = lane i; 96 bits).
    // SR variants add an opaque u32 seed: stochastic rounding is rounding plus a
    // randomness source, and FPSan's cast already truncates deterministically, so
    // the payload-domain answer equals the non-SR pack (same model as every other
    // SR cvt above). Each block is __has_builtin-gated (gfx1250-exclusive).
    // =============================================================================
    using v8f_native   = float __attribute__((ext_vector_type(8)));
    using v8h_native   = _Float16 __attribute__((ext_vector_type(8)));
    using v8bf_native  = __bf16 __attribute__((ext_vector_type(8)));
    using v16h_native  = _Float16 __attribute__((ext_vector_type(16)));
    using v16bf_native = __bf16 __attribute__((ext_vector_type(16)));
    using v2u32_native = unsigned __attribute__((ext_vector_type(2)));
    using v3u32_native = unsigned __attribute__((ext_vector_type(3)));

    namespace detail
    {
        // Assemble 8 fp8/bf8 payload bytes into a contiguous v2u32 (byte i = b[i],
        // little-endian: word w holds bytes 4w..4w+3).
        FPSAN_DEVICE inline v2u32_native pack8_bytes(const std::uint8_t b[8])
        {
            v2u32_native out{};
            for(int w = 0; w < 2; ++w)
            {
                std::uint32_t word = 0;
                for(int k = 0; k < 4; ++k)
                    word |= static_cast<std::uint32_t>(b[4 * w + k]) << (8 * k);
                out[w] = word;
            }
            return out;
        }
        // Assemble 8 fp4 payload nibbles into one u32 (nibble i = n[i] & 0xF).
        FPSAN_DEVICE inline std::uint32_t pack8_nibbles(const std::uint8_t n[8])
        {
            std::uint32_t out = 0;
            for(int i = 0; i < 8; ++i)
                out |= (static_cast<std::uint32_t>(n[i]) & 0xFu) << (4 * i);
            return out;
        }
    } // namespace detail

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_pk8_fp8_f32)
    // ---- pk8 -> fp8/bf8: 8 lanes (f32/f16/bf16) -> v2u32 (8 bytes), each /scale.
#define FPSAN_DEFINE_CVT_SCALEF32_PK8_FP8(NAME, FP8, VEC, BUILTIN)                           \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>        \
    FPSAN_DEVICE v2u32_native NAME(Value<VEC, S, C> v, Value<float, S, C> scale)             \
    {                                                                                        \
        if constexpr(S == Semantics::Native)                                                 \
            return BUILTIN(v.to_float(), scale.to_float());                                  \
        else                                                                                 \
        {                                                                                    \
            std::uint8_t b[8];                                                               \
            for(int i = 0; i < 8; ++i)                                                       \
                b[i] = static_cast<std::uint8_t>(                                            \
                    fpsan::cast<FP8>(fpsan::cast<float>(v.get(i)) / scale).fpsan_payload()); \
            return detail::pack8_bytes(b);                                                   \
        }                                                                                    \
    }
    FPSAN_DEFINE_CVT_SCALEF32_PK8_FP8(amdgcn_cvt_scalef32_pk8_fp8_f32,
                                      fp8_e4m3,
                                      v8f_native,
                                      __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32)
    FPSAN_DEFINE_CVT_SCALEF32_PK8_FP8(amdgcn_cvt_scalef32_pk8_bf8_f32,
                                      fp8_e5m2,
                                      v8f_native,
                                      __builtin_amdgcn_cvt_scalef32_pk8_bf8_f32)
    FPSAN_DEFINE_CVT_SCALEF32_PK8_FP8(amdgcn_cvt_scalef32_pk8_fp8_f16,
                                      fp8_e4m3,
                                      v8h_native,
                                      __builtin_amdgcn_cvt_scalef32_pk8_fp8_f16)
    FPSAN_DEFINE_CVT_SCALEF32_PK8_FP8(amdgcn_cvt_scalef32_pk8_bf8_f16,
                                      fp8_e5m2,
                                      v8h_native,
                                      __builtin_amdgcn_cvt_scalef32_pk8_bf8_f16)
    FPSAN_DEFINE_CVT_SCALEF32_PK8_FP8(amdgcn_cvt_scalef32_pk8_fp8_bf16,
                                      fp8_e4m3,
                                      v8bf_native,
                                      __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16)
    FPSAN_DEFINE_CVT_SCALEF32_PK8_FP8(amdgcn_cvt_scalef32_pk8_bf8_bf16,
                                      fp8_e5m2,
                                      v8bf_native,
                                      __builtin_amdgcn_cvt_scalef32_pk8_bf8_bf16)
#undef FPSAN_DEFINE_CVT_SCALEF32_PK8_FP8

    // ---- pk8 -> fp4: 8 lanes -> u32 (8 nibbles), each /scale.
#define FPSAN_DEFINE_CVT_SCALEF32_PK8_FP4(NAME, VEC, BUILTIN)                               \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>       \
    FPSAN_DEVICE std::uint32_t NAME(Value<VEC, S, C> v, Value<float, S, C> scale)           \
    {                                                                                       \
        if constexpr(S == Semantics::Native)                                                \
            return BUILTIN(v.to_float(), scale.to_float());                                 \
        else                                                                                \
        {                                                                                   \
            std::uint8_t n[8];                                                              \
            for(int i = 0; i < 8; ++i)                                                      \
                n[i] = static_cast<std::uint8_t>(                                           \
                    detail::f32_to_fp4_nibble<S, C>(fpsan::cast<float>(v.get(i)) / scale)); \
            return detail::pack8_nibbles(n);                                                \
        }                                                                                   \
    }
    FPSAN_DEFINE_CVT_SCALEF32_PK8_FP4(amdgcn_cvt_scalef32_pk8_fp4_f32,
                                      v8f_native,
                                      __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32)
    FPSAN_DEFINE_CVT_SCALEF32_PK8_FP4(amdgcn_cvt_scalef32_pk8_fp4_f16,
                                      v8h_native,
                                      __builtin_amdgcn_cvt_scalef32_pk8_fp4_f16)
    FPSAN_DEFINE_CVT_SCALEF32_PK8_FP4(amdgcn_cvt_scalef32_pk8_fp4_bf16,
                                      v8bf_native,
                                      __builtin_amdgcn_cvt_scalef32_pk8_fp4_bf16)
#undef FPSAN_DEFINE_CVT_SCALEF32_PK8_FP4

    // ---- pk16 -> fp6/bf6: 16 lanes -> v3u32 (16 6-bit codes, contiguous), each
    // /scale. fp6 vs bf6 differ only in the Float-mode builtin (the FPSan payload
    // narrow is a width-6 truncate either way), exactly like the gfx950 pk32 pack.
#define FPSAN_DEFINE_CVT_SCALEF32_PK16_FP6(NAME, VEC, BUILTIN)                                \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>         \
    FPSAN_DEVICE v3u32_native NAME(Value<VEC, S, C> v, Value<float, S, C> scale)              \
    {                                                                                         \
        if constexpr(S == Semantics::Native)                                                  \
            return BUILTIN(v.to_float(), scale.to_float());                                   \
        else                                                                                  \
        {                                                                                     \
            v3u32_native out{};                                                               \
            for(int i = 0; i < 16; ++i)                                                       \
                detail::insert6(                                                              \
                    out, i, detail::f32_to_sub6<S, C>(fpsan::cast<float>(v.get(i)) / scale)); \
            return out;                                                                       \
        }                                                                                     \
    }
    FPSAN_DEFINE_CVT_SCALEF32_PK16_FP6(amdgcn_cvt_scalef32_pk16_fp6_f32,
                                       v16f_native_cvt,
                                       __builtin_amdgcn_cvt_scalef32_pk16_fp6_f32)
    FPSAN_DEFINE_CVT_SCALEF32_PK16_FP6(amdgcn_cvt_scalef32_pk16_bf6_f32,
                                       v16f_native_cvt,
                                       __builtin_amdgcn_cvt_scalef32_pk16_bf6_f32)
    FPSAN_DEFINE_CVT_SCALEF32_PK16_FP6(amdgcn_cvt_scalef32_pk16_fp6_f16,
                                       v16h_native,
                                       __builtin_amdgcn_cvt_scalef32_pk16_fp6_f16)
    FPSAN_DEFINE_CVT_SCALEF32_PK16_FP6(amdgcn_cvt_scalef32_pk16_bf6_f16,
                                       v16h_native,
                                       __builtin_amdgcn_cvt_scalef32_pk16_bf6_f16)
    FPSAN_DEFINE_CVT_SCALEF32_PK16_FP6(amdgcn_cvt_scalef32_pk16_fp6_bf16,
                                       v16bf_native,
                                       __builtin_amdgcn_cvt_scalef32_pk16_fp6_bf16)
    FPSAN_DEFINE_CVT_SCALEF32_PK16_FP6(amdgcn_cvt_scalef32_pk16_bf6_bf16,
                                       v16bf_native,
                                       __builtin_amdgcn_cvt_scalef32_pk16_bf6_bf16)
#undef FPSAN_DEFINE_CVT_SCALEF32_PK16_FP6

    // ---- SR pk8 -> fp8/bf8 (seed opaque to the payload-domain answer).
#define FPSAN_DEFINE_CVT_SCALEF32_SR_PK8_FP8(NAME, FP8, VEC, BUILTIN)                        \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>        \
    FPSAN_DEVICE v2u32_native NAME(                                                          \
        Value<VEC, S, C> v, std::uint32_t seed, Value<float, S, C> scale)                    \
    {                                                                                        \
        if constexpr(S == Semantics::Native)                                                 \
            return BUILTIN(v.to_float(), seed, scale.to_float());                            \
        else                                                                                 \
        {                                                                                    \
            (void)seed;                                                                      \
            std::uint8_t b[8];                                                               \
            for(int i = 0; i < 8; ++i)                                                       \
                b[i] = static_cast<std::uint8_t>(                                            \
                    fpsan::cast<FP8>(fpsan::cast<float>(v.get(i)) / scale).fpsan_payload()); \
            return detail::pack8_bytes(b);                                                   \
        }                                                                                    \
    }
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK8_FP8(amdgcn_cvt_scalef32_sr_pk8_fp8_f32,
                                         fp8_e4m3,
                                         v8f_native,
                                         __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f32)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK8_FP8(amdgcn_cvt_scalef32_sr_pk8_bf8_f32,
                                         fp8_e5m2,
                                         v8f_native,
                                         __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_f32)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK8_FP8(amdgcn_cvt_scalef32_sr_pk8_fp8_f16,
                                         fp8_e4m3,
                                         v8h_native,
                                         __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK8_FP8(amdgcn_cvt_scalef32_sr_pk8_bf8_f16,
                                         fp8_e5m2,
                                         v8h_native,
                                         __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_f16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK8_FP8(amdgcn_cvt_scalef32_sr_pk8_fp8_bf16,
                                         fp8_e4m3,
                                         v8bf_native,
                                         __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_bf16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK8_FP8(amdgcn_cvt_scalef32_sr_pk8_bf8_bf16,
                                         fp8_e5m2,
                                         v8bf_native,
                                         __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_bf16)
#undef FPSAN_DEFINE_CVT_SCALEF32_SR_PK8_FP8

    // ---- SR pk8 -> fp4.
#define FPSAN_DEFINE_CVT_SCALEF32_SR_PK8_FP4(NAME, VEC, BUILTIN)                            \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>       \
    FPSAN_DEVICE std::uint32_t NAME(                                                        \
        Value<VEC, S, C> v, std::uint32_t seed, Value<float, S, C> scale)                   \
    {                                                                                       \
        if constexpr(S == Semantics::Native)                                                \
            return BUILTIN(v.to_float(), seed, scale.to_float());                           \
        else                                                                                \
        {                                                                                   \
            (void)seed;                                                                     \
            std::uint8_t n[8];                                                              \
            for(int i = 0; i < 8; ++i)                                                      \
                n[i] = static_cast<std::uint8_t>(                                           \
                    detail::f32_to_fp4_nibble<S, C>(fpsan::cast<float>(v.get(i)) / scale)); \
            return detail::pack8_nibbles(n);                                                \
        }                                                                                   \
    }
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK8_FP4(amdgcn_cvt_scalef32_sr_pk8_fp4_f32,
                                         v8f_native,
                                         __builtin_amdgcn_cvt_scalef32_sr_pk8_fp4_f32)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK8_FP4(amdgcn_cvt_scalef32_sr_pk8_fp4_f16,
                                         v8h_native,
                                         __builtin_amdgcn_cvt_scalef32_sr_pk8_fp4_f16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK8_FP4(amdgcn_cvt_scalef32_sr_pk8_fp4_bf16,
                                         v8bf_native,
                                         __builtin_amdgcn_cvt_scalef32_sr_pk8_fp4_bf16)
#undef FPSAN_DEFINE_CVT_SCALEF32_SR_PK8_FP4

    // ---- SR pk16 -> fp6/bf6.
#define FPSAN_DEFINE_CVT_SCALEF32_SR_PK16_FP6(NAME, VEC, BUILTIN)                             \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>         \
    FPSAN_DEVICE v3u32_native NAME(                                                           \
        Value<VEC, S, C> v, std::uint32_t seed, Value<float, S, C> scale)                     \
    {                                                                                         \
        if constexpr(S == Semantics::Native)                                                  \
            return BUILTIN(v.to_float(), seed, scale.to_float());                             \
        else                                                                                  \
        {                                                                                     \
            (void)seed;                                                                       \
            v3u32_native out{};                                                               \
            for(int i = 0; i < 16; ++i)                                                       \
                detail::insert6(                                                              \
                    out, i, detail::f32_to_sub6<S, C>(fpsan::cast<float>(v.get(i)) / scale)); \
            return out;                                                                       \
        }                                                                                     \
    }
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK16_FP6(amdgcn_cvt_scalef32_sr_pk16_fp6_f32,
                                          v16f_native_cvt,
                                          __builtin_amdgcn_cvt_scalef32_sr_pk16_fp6_f32)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK16_FP6(amdgcn_cvt_scalef32_sr_pk16_bf6_f32,
                                          v16f_native_cvt,
                                          __builtin_amdgcn_cvt_scalef32_sr_pk16_bf6_f32)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK16_FP6(amdgcn_cvt_scalef32_sr_pk16_fp6_f16,
                                          v16h_native,
                                          __builtin_amdgcn_cvt_scalef32_sr_pk16_fp6_f16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK16_FP6(amdgcn_cvt_scalef32_sr_pk16_bf6_f16,
                                          v16h_native,
                                          __builtin_amdgcn_cvt_scalef32_sr_pk16_bf6_f16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK16_FP6(amdgcn_cvt_scalef32_sr_pk16_fp6_bf16,
                                          v16bf_native,
                                          __builtin_amdgcn_cvt_scalef32_sr_pk16_fp6_bf16)
    FPSAN_DEFINE_CVT_SCALEF32_SR_PK16_FP6(amdgcn_cvt_scalef32_sr_pk16_bf6_bf16,
                                          v16bf_native,
                                          __builtin_amdgcn_cvt_scalef32_sr_pk16_bf6_bf16)
#undef FPSAN_DEFINE_CVT_SCALEF32_SR_PK16_FP6
#endif // __has_builtin(cvt_scalef32_pk8_fp8_f32)

    // =============================================================================
    // gfx1250 E5M3 fp8 conversions ("fp8e5m3-insts").
    //
    // E5M3 is gfx1250's higher-precision 8-bit scale format: UNSIGNED, 5 exponent
    // bits (bias 15), 3 mantissa bits, NO sign bit, no infinity, NaN == 0xFF only,
    // 0x00 == zero, denormals supported (it
    // is NOT in LLVM APFloat). The hardware reaches it by OVERLOADING the ordinary
    // f32<->fp8 convert opcodes via the CLAMP bit -- exposed in LLVM/Clang as the
    // dedicated *_e5m3 builtins whose signatures are byte-identical to the
    // non-e5m3 cvt_{f32_fp8,pk_fp8_f32,sr_fp8_f32}. Float mode forwards to the
    // builtin; the tests assert against the HOST E5M3 codec (detail::kFp8E5M3),
    // which is the authoritative reference.
    //
    // FPSan model: E5M3 cannot be a Value<> element type (1 + 5 + 3 != 8, since it
    // has no sign bit), so there is no Value<fp8_e5m3>. The payload ring only ever
    // cares about STORAGE WIDTH for a resize, and that is 8 bits -- identical to
    // e4m3/e5m2. So the FPSan path is the same width-8 deterministic resize as its
    // siblings (decode = subbyte_widen<8>; encode = low 8 payload bits), keeping
    // mixed e4m3/e5m2/e5m3 FPSan kernels uniform; the format's unsignedness is only
    // modeled in (authoritative) Float mode.
    // =============================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_f32_fp8_e5m3)
    // Decode: byte ByteIdx of `packed`, interpreted as E5M3, -> f32.
    template <int ByteIdx, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<float, S, C> amdgcn_cvt_f32_fp8_e5m3(int packed)
    {
        if constexpr(S == Semantics::Native)
            return Value<float, S, C>(__builtin_amdgcn_cvt_f32_fp8_e5m3(packed, ByteIdx));
        else
        {
            const std::uint8_t byte = static_cast<std::uint8_t>(
                (static_cast<std::uint32_t>(packed) >> (ByteIdx * 8)) & 0xFFu);
            return detail::subbyte_widen<8, S, C>(byte);
        }
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_pk_fp8_f32_e5m3)
    // Pack: two f32 -> two E5M3 bytes spliced into the lo (DstLo=true) or hi half
    // of `old`. Mirrors cvt_pk_fp8_f32: the builtin word-select is false=low, so
    // pass !DstLo to keep the Float and FPSan paths placing the bytes in the same
    // half (a -> low byte of the half, b -> high byte).
    template <bool DstLo, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE int amdgcn_cvt_pk_fp8_f32_e5m3(Value<float, S, C> a, Value<float, S, C> b, int old)
    {
        if constexpr(S == Semantics::Native)
            return __builtin_amdgcn_cvt_pk_fp8_f32_e5m3(a.to_float(), b.to_float(), old, !DstLo);
        else
            return detail::pack_fp8_pair<DstLo>(static_cast<std::uint8_t>(a.fpsan_payload()),
                                                static_cast<std::uint8_t>(b.fpsan_payload()),
                                                old);
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_sr_fp8_f32_e5m3)
    // SR pack: stochastic-round one f32 to an E5M3 byte at position ByteIdx of
    // `packed_old`. The clang builtin signature is (val, seed, old, byte_sel) --
    // confirmed by clang/test/CodeGenOpenCL/builtins-amdgcn-gfx1250.cl, which
    // lowers __builtin_amdgcn_cvt_sr_fp8_f32_e5m3(a, b, old, 3) to
    // llvm.amdgcn.cvt.sr.fp8.f32.e5m3(val, seed=b, old, 3) with no reordering.
    // seed is opaque to the FPSan deterministic-truncate answer.
    template <int ByteIdx, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE int
        amdgcn_cvt_sr_fp8_f32_e5m3(Value<float, S, C> val, int packed_old, std::uint32_t seed)
    {
        if constexpr(S == Semantics::Native)
            return __builtin_amdgcn_cvt_sr_fp8_f32_e5m3(
                val.to_float(), static_cast<int>(seed), packed_old, ByteIdx);
        else
        {
            (void)seed;
            return detail::splice_fp8_byte<ByteIdx>(static_cast<std::uint8_t>(val.fpsan_payload()),
                                                    packed_old);
        }
    }
#endif

    // =============================================================================
    // gfx1250 WIDE block-scale UNPACK family (cvt_scale_pk8 / cvt_scale_pk16).
    //
    // The decode siblings of the cvt_scalef32_pk8/pk16 PACK family: read a packed
    // stream of 8 (pk8) or 16 (pk16) narrow codes and widen each to f32/f16/bf16,
    // multiplying by a per-lane block scale. Unlike the gfx950 cvt_scalef32_*
    // family (whose scale is a plain f32 operand), here the scale is an E8M0 byte
    // selected from a packed u32 `scale` operand by the _Constant `ScaleSel`
    // (opsel): the hardware picks one byte per 16-lane half (Block16/Block32 mode)
    // and decodes it as 2^(code-127). The exact per-lane byte selection is
    // silicon-grounded.
    //
    //   pk8  fp8/bf8 : v2u32 (8 bytes,   byte i  = code i) -> v8  (f32/f16/bf16).
    //   pk8  fp4     : u32   (8 nibbles,  nib  i  = code i) -> v8.
    //   pk16 fp6/bf6 : v3u32 (16 codes,   contiguous 6-bit) -> v16.
    //
    // Float mode forwards to the hardware builtin (authoritative). FPSan mode is a
    // plain payload WIDEN of each narrow code (detail::subbyte_widen<8/6/4>) with
    // NO scale applied: the block scale is a magnitude-only Float-domain effect,
    // and the payload ring tracks precision/width, not the E8M0 multiply
    // (Float-only -- a single-lane payload model cannot reproduce the
    // per-lane byte selection anyway). Tests pin the Float scale operand to
    // all-0x7f (E8M0 127 == x1) so Float and FPSan agree on an exact decode, plus
    // a separate Float-only case for a 2^n scale.
    // =============================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scale_pk8_f32_fp8)
    // ---- pk8 unpack from fp8/bf8: v2u32 (8 bytes) -> v8, each * scale (width-8).
#define FPSAN_DEFINE_CVT_SCALE_UNPACK_PK8_FP8(NAME, DstFT, VEC, BUILTIN)             \
    template <int         ScaleSel,                                                  \
              Semantics   S = Semantics::Native,                                     \
              Conversions C = Conversions::Explicit>                                 \
    FPSAN_DEVICE Value<VEC, S, C> NAME(v2u32_native packed, std::uint32_t scale)     \
    {                                                                                \
        if constexpr(S == Semantics::Native)                                         \
            return Value<VEC, S, C>(BUILTIN(packed, scale, ScaleSel));               \
        else                                                                         \
        {                                                                            \
            (void)scale;                                                             \
            Value<VEC, S, C> r{};                                                    \
            for(int i = 0; i < 8; ++i)                                               \
            {                                                                        \
                const std::uint32_t byte = (packed[i / 4] >> (8 * (i % 4))) & 0xFFu; \
                r.set(i, fpsan::cast<DstFT>(detail::subbyte_widen<8, S, C>(byte)));  \
            }                                                                        \
            return r;                                                                \
        }                                                                            \
    }
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK8_FP8(amdgcn_cvt_scale_pk8_f32_fp8,
                                          float,
                                          v8f_native,
                                          __builtin_amdgcn_cvt_scale_pk8_f32_fp8)
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK8_FP8(amdgcn_cvt_scale_pk8_f32_bf8,
                                          float,
                                          v8f_native,
                                          __builtin_amdgcn_cvt_scale_pk8_f32_bf8)
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK8_FP8(amdgcn_cvt_scale_pk8_f16_fp8,
                                          _Float16,
                                          v8h_native,
                                          __builtin_amdgcn_cvt_scale_pk8_f16_fp8)
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK8_FP8(amdgcn_cvt_scale_pk8_f16_bf8,
                                          _Float16,
                                          v8h_native,
                                          __builtin_amdgcn_cvt_scale_pk8_f16_bf8)
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK8_FP8(amdgcn_cvt_scale_pk8_bf16_fp8,
                                          __bf16,
                                          v8bf_native,
                                          __builtin_amdgcn_cvt_scale_pk8_bf16_fp8)
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK8_FP8(amdgcn_cvt_scale_pk8_bf16_bf8,
                                          __bf16,
                                          v8bf_native,
                                          __builtin_amdgcn_cvt_scale_pk8_bf16_bf8)
#undef FPSAN_DEFINE_CVT_SCALE_UNPACK_PK8_FP8

    // ---- pk8 unpack from fp4: u32 (8 nibbles) -> v8, each * scale (width-4).
#define FPSAN_DEFINE_CVT_SCALE_UNPACK_PK8_FP4(NAME, DstFT, VEC, BUILTIN)           \
    template <int         ScaleSel,                                                \
              Semantics   S = Semantics::Native,                                   \
              Conversions C = Conversions::Explicit>                               \
    FPSAN_DEVICE Value<VEC, S, C> NAME(std::uint32_t packed, std::uint32_t scale)  \
    {                                                                              \
        if constexpr(S == Semantics::Native)                                       \
            return Value<VEC, S, C>(BUILTIN(packed, scale, ScaleSel));             \
        else                                                                       \
        {                                                                          \
            (void)scale;                                                           \
            Value<VEC, S, C> r{};                                                  \
            for(int i = 0; i < 8; ++i)                                             \
            {                                                                      \
                const std::uint32_t nib = (packed >> (4 * i)) & 0xFu;              \
                r.set(i, fpsan::cast<DstFT>(detail::subbyte_widen<4, S, C>(nib))); \
            }                                                                      \
            return r;                                                              \
        }                                                                          \
    }
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK8_FP4(amdgcn_cvt_scale_pk8_f32_fp4,
                                          float,
                                          v8f_native,
                                          __builtin_amdgcn_cvt_scale_pk8_f32_fp4)
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK8_FP4(amdgcn_cvt_scale_pk8_f16_fp4,
                                          _Float16,
                                          v8h_native,
                                          __builtin_amdgcn_cvt_scale_pk8_f16_fp4)
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK8_FP4(amdgcn_cvt_scale_pk8_bf16_fp4,
                                          __bf16,
                                          v8bf_native,
                                          __builtin_amdgcn_cvt_scale_pk8_bf16_fp4)
#undef FPSAN_DEFINE_CVT_SCALE_UNPACK_PK8_FP4

    // ---- pk16 unpack from fp6/bf6: v3u32 (16 contiguous 6-bit codes) -> v16, each
    // * scale (width-6). fp6 vs bf6 differ only in the Float-mode builtin; the
    // FPSan payload widen is a width-6 sign-resize either way.
#define FPSAN_DEFINE_CVT_SCALE_UNPACK_PK16_FP6(NAME, DstFT, VEC, BUILTIN)                \
    template <int         ScaleSel,                                                      \
              Semantics   S = Semantics::Native,                                         \
              Conversions C = Conversions::Explicit>                                     \
    FPSAN_DEVICE Value<VEC, S, C> NAME(v3u32_native packed, std::uint32_t scale)         \
    {                                                                                    \
        if constexpr(S == Semantics::Native)                                             \
            return Value<VEC, S, C>(BUILTIN(packed, scale, ScaleSel));                   \
        else                                                                             \
        {                                                                                \
            (void)scale;                                                                 \
            Value<VEC, S, C> r{};                                                        \
            for(int i = 0; i < 16; ++i)                                                  \
            {                                                                            \
                const int     p = i * 6, wi = p >> 5, off = p & 31;                      \
                std::uint32_t f = packed[wi] >> off;                                     \
                if(off > 26)                                                             \
                    f |= packed[wi + 1] << (32 - off);                                   \
                r.set(i, fpsan::cast<DstFT>(detail::subbyte_widen<6, S, C>(f & 0x3Fu))); \
            }                                                                            \
            return r;                                                                    \
        }                                                                                \
    }
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK16_FP6(amdgcn_cvt_scale_pk16_f32_fp6,
                                           float,
                                           v16f_native_cvt,
                                           __builtin_amdgcn_cvt_scale_pk16_f32_fp6)
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK16_FP6(amdgcn_cvt_scale_pk16_f32_bf6,
                                           float,
                                           v16f_native_cvt,
                                           __builtin_amdgcn_cvt_scale_pk16_f32_bf6)
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK16_FP6(amdgcn_cvt_scale_pk16_f16_fp6,
                                           _Float16,
                                           v16h_native,
                                           __builtin_amdgcn_cvt_scale_pk16_f16_fp6)
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK16_FP6(amdgcn_cvt_scale_pk16_f16_bf6,
                                           _Float16,
                                           v16h_native,
                                           __builtin_amdgcn_cvt_scale_pk16_f16_bf6)
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK16_FP6(amdgcn_cvt_scale_pk16_bf16_fp6,
                                           __bf16,
                                           v16bf_native,
                                           __builtin_amdgcn_cvt_scale_pk16_bf16_fp6)
    FPSAN_DEFINE_CVT_SCALE_UNPACK_PK16_FP6(amdgcn_cvt_scale_pk16_bf16_bf6,
                                           __bf16,
                                           v16bf_native,
                                           __builtin_amdgcn_cvt_scale_pk16_bf16_bf6)
#undef FPSAN_DEFINE_CVT_SCALE_UNPACK_PK16_FP6
#endif // __has_builtin(__builtin_amdgcn_cvt_scale_pk8_f32_fp8)

} // namespace fpsan

#endif // FPSAN_AMDGCN_CVT_HPP
