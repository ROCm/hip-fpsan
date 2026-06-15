// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/detail/traits.hpp
// ----------------------------------------------------------------------------
// Compile-time description of an IEEE-754-style binary floating-point type:
// its bit width, the unsigned integer of the same size, and the
// sign/exponent/mantissa split.
//
// std::numeric_limits is intentionally NOT used: on the toolchains we target it
// is not specialized for _Float16 / __bf16, so we hand-write the (standardized)
// layouts. Each trait is checked for internal consistency with a static_assert.
// ----------------------------------------------------------------------------
#ifndef FPSAN_DETAIL_TRAITS_HPP
#define FPSAN_DETAIL_TRAITS_HPP

#include "fpsan/detail/config.hpp"

#include <cstdint>
#include <type_traits>

namespace fpsan
{
    namespace detail
    {

        template <class...>
        inline constexpr bool always_false = false;

        template <unsigned Bytes>
        struct uint_by_size;
        template <>
        struct uint_by_size<1>
        {
            using type = std::uint8_t;
        };
        template <>
        struct uint_by_size<2>
        {
            using type = std::uint16_t;
        };
        template <>
        struct uint_by_size<4>
        {
            using type = std::uint32_t;
        };
        template <>
        struct uint_by_size<8>
        {
            using type = std::uint64_t;
        };

        // Primary template is intentionally undefined: using Value with an
        // unsupported underlying type is a compile error pointing here.
        template <class T>
        struct fp_traits
        {
            static_assert(always_false<T>,
                          "fpsan: unsupported floating-point type. Supported: float, "
                          "double, _Float16, __bf16 (the latter two where available).");
        };

#define FPSAN_DEFINE_FP_TRAITS(T, MANT, EXP, BIAS)                                               \
    template <>                                                                                  \
    struct fp_traits<T>                                                                          \
    {                                                                                            \
        using value_type                        = T;                                             \
        using bits_type                         = typename uint_by_size<sizeof(T)>::type;        \
        static constexpr unsigned bit_width     = sizeof(T) * 8;                                 \
        static constexpr unsigned mantissa_bits = (MANT);                                        \
        static constexpr unsigned exponent_bits = (EXP);                                         \
        static constexpr int      bias          = (BIAS);                                        \
        static_assert(1 + (EXP) + (MANT) == bit_width, "fpsan: inconsistent fp_traits for " #T); \
    }

        FPSAN_DEFINE_FP_TRAITS(float, 23, 8, 127);
        FPSAN_DEFINE_FP_TRAITS(double, 52, 11, 1023);

// Extended-float availability (FPSAN_HAS_FLOAT16 / FPSAN_HAS_BF16) is derived
// *only* from compiler-predefined macros, never from a user-settable knob. These
// flags decide which fp_traits<> specializations exist, so two translation units
// that disagreed about them would form different definitions of the same
// inline/template entities -- a One-Definition-Rule violation (the same class of
// bug as a per-TU `-mavx512f`). Keying them strictly off the toolchain makes
// every TU built with the same compiler+target agree. They are therefore not
// overridable; a hand-defined value is rejected here.
#if defined(FPSAN_HAS_FLOAT16) || defined(FPSAN_HAS_BF16)
#error \
    "fpsan: FPSAN_HAS_FLOAT16 / FPSAN_HAS_BF16 are auto-detected from the compiler and must not be defined manually (an inconsistent value across translation units would be an ODR violation)."
#endif

// _Float16 (IEEE binary16, E5M10): Clang and GCC predefine __FLT16_MANT_DIG__
// exactly when the type is usable on the target.
#if defined(__FLT16_MANT_DIG__)
#define FPSAN_HAS_FLOAT16 1
#else
#define FPSAN_HAS_FLOAT16 0
#endif

// __bf16 (bfloat16, E8M7): there is no dedicated predefined macro. On Clang,
// __is_identifier(__bf16) is 0 precisely when __bf16 is a built-in type keyword
// for the target (true on host x86/AArch64 and on the amdgcn device pass). On
// GCC the type exists from GCC 13 on the targets that have it; detect that via
// the predefined version/arch macros.
#if defined(__is_identifier)
#if !__is_identifier(__bf16)
#define FPSAN_HAS_BF16 1
#else
#define FPSAN_HAS_BF16 0
#endif
#elif defined(__GNUC__) && (__GNUC__ >= 13) && (defined(__x86_64__) || defined(__aarch64__))
#define FPSAN_HAS_BF16 1
#else
#define FPSAN_HAS_BF16 0
#endif

#if FPSAN_HAS_FLOAT16
        FPSAN_DEFINE_FP_TRAITS(_Float16, 10, 5, 15);
#endif
#if FPSAN_HAS_BF16
        FPSAN_DEFINE_FP_TRAITS(__bf16, 7, 8, 127);
#endif

        // True when fp_traits<T> is a usable (defined) specialization.
        template <class T, class = void>
        inline constexpr bool is_supported_float_v = false;
        template <class T>
        inline constexpr bool is_supported_float_v<T, std::void_t<typename fp_traits<T>::bits_type>>
            = true;

        // ---------------------------------------------------------------------------
        //  Vector element types.
        //
        //  Value<T> also supports T being a Clang/GCC vector of a supported float, e.g.
        //  `float __attribute__((ext_vector_type(8)))`. Such a Value is a SIMD bundle
        //  of per-lane FPSan values: the payload algebra is applied lane-wise, which
        //  the native integer-vector operators already do (masking keeps each lane
        //  exact). value_traits<T> unifies the scalar and vector cases.
        // ---------------------------------------------------------------------------

        // A Clang/GCC vector is detected by having a subscript operator (scalars and
        // our class types do not).
        template <class T, class = void>
        inline constexpr bool is_clang_vector_v = false;
        template <class T>
        inline constexpr bool is_clang_vector_v<T, std::void_t<decltype(std::declval<T&>()[0])>>
            = true;

        template <class T>
        using vector_element_t
            = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<T&>()[0])>>;

        // Build an unsigned vector type with `Bytes` total size and the given lane.
        template <class LaneBits, unsigned long Bytes>
        using uint_vector_t = LaneBits __attribute__((__vector_size__(Bytes)));

        // Lane (scalar) type of a possibly-vector bits type.
        template <class B, bool = is_clang_vector_v<B>>
        struct bits_lane
        {
            using type = B;
        };
        template <class B>
        struct bits_lane<B, true>
        {
            using type = vector_element_t<B>;
        };
        template <class B>
        using bits_lane_t = typename bits_lane<B>::type;

        // Signed counterpart of a (possibly vector) unsigned bits type, used for the
        // signed-payload ordering in FPSan mode. Specialized so make_signed is never
        // applied to a vector type directly.
        template <class B, bool = is_clang_vector_v<B>>
        struct signed_bits
        {
            using type = std::make_signed_t<B>;
        };
        template <class B>
        struct signed_bits<B, true>
        {
            using type = uint_vector_t<std::make_signed_t<vector_element_t<B>>, sizeof(B)>;
        };
        template <class B>
        using signed_bits_t = typename signed_bits<B>::type;

        template <class T, bool = is_clang_vector_v<T>>
        struct value_traits
        {
            // Scalar case.
            using element_type                  = T;
            using element_traits                = fp_traits<T>;
            using bits_type                     = typename fp_traits<T>::bits_type;
            static constexpr unsigned lanes     = 1;
            static constexpr bool     is_vector = false;
        };
        template <class T>
        struct value_traits<T, true>
        {
            // Vector case: the per-lane element is the vector's element type.
            using element_type   = vector_element_t<T>;
            using element_traits = fp_traits<element_type>;
            using bits_type = uint_vector_t<typename fp_traits<element_type>::bits_type, sizeof(T)>;
            static constexpr unsigned lanes     = sizeof(T) / sizeof(element_type);
            static constexpr bool     is_vector = true;
        };

    } // namespace detail
} // namespace fpsan

#endif // FPSAN_DETAIL_TRAITS_HPP
