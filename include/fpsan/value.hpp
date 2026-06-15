// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/value.hpp
// ----------------------------------------------------------------------------
// Value<float_type, semantics, conversions>
//
//   float_type   underlying real type: a supported scalar (float, double,
//                _Float16, __bf16) OR a Clang/GCC vector of one of those, e.g.
//                `float __attribute__((ext_vector_type(8)))`.
//   semantics    Semantics::Native : native arithmetic of float_type (drop-in)
//                Semantics::Triton : Triton-style FPSan integer-payload
//                arithmetic
//   conversions  Conversions::Implicit / Conversions::Explicit
//
// A vector float_type makes Value a SIMD bundle: the payload algebra is applied
// lane-wise (the native integer-vector operators do exactly that, since each op
// masks to the lane width). Comparisons then yield a per-lane mask, mirroring
// native vector comparisons.
// ----------------------------------------------------------------------------
#ifndef FPSAN_VALUE_HPP
#define FPSAN_VALUE_HPP

#include "fpsan/detail/config.hpp"
#include "fpsan/detail/mix.hpp"
#include "fpsan/detail/traits.hpp"

#include <cstdint>
#include <type_traits>

namespace fpsan
{

    enum class Semantics
    {
        Native, // the native float; no sanitization
        Triton, // Triton-style scrambling payload in the free ring Z/2^w

        // Deprecated former spellings, kept as value-preserving aliases so old
        // code still compiles (with a warning). Prefer the names above.
        Float [[deprecated("Semantics::Float was renamed to Semantics::Native")]] = Native,
        FPSan [[deprecated("Semantics::FPSan was renamed to Semantics::Triton")]] = Triton,
    };
    enum class Conversions
    {
        Implicit,
        Explicit
    };

    template <class float_type_, Semantics semantics_, Conversions conversions_>
    class Value
    {
        using vt = detail::value_traits<float_type_>;

    public:
        using float_type   = float_type_;
        using element_type = typename vt::element_type; // scalar lane float type
        static_assert(detail::is_supported_float_v<element_type>,
                      "fpsan: Value's element type must be a supported floating-"
                      "point type (float, double, _Float16, __bf16, fp8_e4m3, "
                      "fp8_e5m2 -- or a vector thereof). Integer matrix intrinsics "
                      "(e.g. WMMA_I32_*_I8) don't fit the Value<float-type> "
                      "framework and are out of scope.");
        using bits_type                     = typename vt::bits_type; // scalar uint or uint-vector
        using signed_bits_type              = detail::signed_bits_t<bits_type>;
        static constexpr unsigned lanes     = vt::lanes;
        static constexpr bool     is_vector = vt::is_vector;

        static constexpr Semantics   semantics   = semantics_;
        static constexpr Conversions conversions = conversions_;

        // Result of a comparison: bool for a scalar, a per-lane mask for a vector
        // (matching native `float_type < float_type`).
        using cmp_t = decltype(std::declval<float_type>() < std::declval<float_type>());

        // In FPSan mode the object IS the integer payload; otherwise it is the float.
        using storage_type
            = std::conditional_t<semantics == Semantics::Triton, bits_type, float_type>;

        // Per-lane mixing configuration (function of the element type only).
        static constexpr detail::MixConfig config = detail::make_mix_config<element_type>();

        // ---- construction / conversion ------------------------------------------
        FPSAN_HOST_DEVICE constexpr Value()
            : storage_{}
        {
        }

        // `conversions` selects whether the float<->Value conversions are implicit or
        // explicit. C++20 expresses this with a single conditional explicit specifier
        // `explicit(conversions == Conversions::Explicit)` on each member; in C++17
        // there is no such specifier, so we provide two overloads -- one plain, one
        // `explicit` -- and use std::enable_if to enable exactly one of them per
        // instantiation. The dummy default template parameter `C = conversions` only
        // exists to make each member a template, which is what lets std::enable_if
        // (SFINAE) discard the unwanted overload. The float->storage conversion is
        // shared via from_float() so the two constructors do not duplicate logic.
        template <Conversions C                                     = conversions,
                  std::enable_if_t<C == Conversions::Implicit, int> = 0>
        FPSAN_HOST_DEVICE constexpr Value(float_type v)
            : storage_(from_float(v))
        {
        }
        template <Conversions C                                     = conversions,
                  std::enable_if_t<C == Conversions::Explicit, int> = 0>
        FPSAN_HOST_DEVICE constexpr explicit Value(float_type v)
            : storage_(from_float(v))
        {
        }

        template <Conversions C                                     = conversions,
                  std::enable_if_t<C == Conversions::Implicit, int> = 0>
        FPSAN_HOST_DEVICE constexpr operator float_type() const
        {
            return to_float();
        }
        template <Conversions C                                     = conversions,
                  std::enable_if_t<C == Conversions::Explicit, int> = 0>
        FPSAN_HOST_DEVICE constexpr explicit operator float_type() const
        {
            return to_float();
        }

        // ---- named accessors -----------------------------------------------------
        FPSAN_HOST_DEVICE constexpr float_type to_float() const
        {
            if constexpr(semantics == Semantics::Triton)
                return unembed(storage_);
            else
                return storage_;
        }

        // The FPSan integer payload (scalar uint, or uint-vector). FPSan mode only.
        FPSAN_HOST_DEVICE constexpr bits_type fpsan_payload() const
        {
            static_assert(semantics == Semantics::Triton,
                          "fpsan_payload() is only defined when semantics == "
                          "Semantics::Triton");
            return storage_;
        }
        FPSAN_HOST_DEVICE static constexpr Value from_fpsan_payload(bits_type p)
        {
            static_assert(semantics == Semantics::Triton,
                          "from_fpsan_payload() is only defined when semantics == "
                          "Semantics::Triton");
            return Value(static_cast<storage_type>(p), raw_tag{});
        }

        // Raw bit pattern of the stored representation (the payload in FPSan mode,
        // the float's bits otherwise) and its inverse. Used for cross-lane data
        // movement (e.g. __shfl), which must move the storage verbatim regardless of
        // mode.
        FPSAN_HOST_DEVICE constexpr bits_type to_storage_bits() const
        {
            if constexpr(semantics == Semantics::Triton)
                return storage_;
            else
                return __builtin_bit_cast(bits_type, storage_);
        }
        FPSAN_HOST_DEVICE static constexpr Value from_storage_bits(bits_type b)
        {
            if constexpr(semantics == Semantics::Triton)
                return raw(static_cast<storage_type>(b));
            else
                return raw(__builtin_bit_cast(float_type, b));
        }

        // Element access for vector Values: lane i as a scalar Value of the element
        // type (and a setter). Lets fragment code index individual lanes.
        FPSAN_HOST_DEVICE constexpr Value<element_type, semantics_, conversions_>
            get(unsigned i) const
        {
            static_assert(is_vector, "get(i) is only defined for vector Values");
            using Scalar = Value<element_type, semantics_, conversions_>;
            if constexpr(semantics == Semantics::Triton)
                return Scalar::from_fpsan_payload(storage_[i]);
            else
                return Scalar(storage_[i]);
        }
        FPSAN_HOST_DEVICE constexpr void set(unsigned                                      i,
                                             Value<element_type, semantics_, conversions_> v)
        {
            static_assert(is_vector, "set(i,v) is only defined for vector Values");
            if constexpr(semantics == Semantics::Triton)
                storage_[i] = v.fpsan_payload();
            else
                storage_[i] = v.to_float();
        }

        // ---- arithmetic ----------------------------------------------------------
        FPSAN_HOST_DEVICE friend constexpr Value operator+(Value a, Value b)
        {
            return a.combine_add(b);
        }
        FPSAN_HOST_DEVICE friend constexpr Value operator-(Value a, Value b)
        {
            return a.combine_sub(b);
        }
        FPSAN_HOST_DEVICE friend constexpr Value operator*(Value a, Value b)
        {
            return a.combine_mul(b);
        }
        FPSAN_HOST_DEVICE friend constexpr Value operator/(Value a, Value b)
        {
            return a.combine_div(b);
        }

        FPSAN_HOST_DEVICE constexpr Value operator+() const
        {
            return *this;
        }
        FPSAN_HOST_DEVICE constexpr Value operator-() const
        {
            if constexpr(semantics == Semantics::Triton)
                return raw(static_cast<storage_type>(detail::ring_neg(config, storage_)));
            else
                return raw(static_cast<storage_type>(-storage_));
        }

        FPSAN_HOST_DEVICE constexpr Value& operator+=(Value o)
        {
            return *this = *this + o;
        }
        FPSAN_HOST_DEVICE constexpr Value& operator-=(Value o)
        {
            return *this = *this - o;
        }
        FPSAN_HOST_DEVICE constexpr Value& operator*=(Value o)
        {
            return *this = *this * o;
        }
        FPSAN_HOST_DEVICE constexpr Value& operator/=(Value o)
        {
            return *this = *this / o;
        }

        // ---- comparisons ---------------------------------------------------------
        // == / != compare payloads in FPSan mode (exact) and the floats otherwise.
        // Ordering in FPSan mode is the *signed integer* order of payloads (Triton's
        // min/max contract), NOT IEEE float order. Vector Values return per-lane
        // masks.
        FPSAN_HOST_DEVICE friend constexpr cmp_t operator==(Value a, Value b)
        {
            return a.eq(b);
        }
        FPSAN_HOST_DEVICE friend constexpr cmp_t operator!=(Value a, Value b)
        {
            return lnot(a.eq(b));
        }
        FPSAN_HOST_DEVICE friend constexpr cmp_t operator<(Value a, Value b)
        {
            return a.less(b);
        }
        FPSAN_HOST_DEVICE friend constexpr cmp_t operator>(Value a, Value b)
        {
            return b.less(a);
        }
        FPSAN_HOST_DEVICE friend constexpr cmp_t operator<=(Value a, Value b)
        {
            return lnot(b.less(a));
        }
        FPSAN_HOST_DEVICE friend constexpr cmp_t operator>=(Value a, Value b)
        {
            return lnot(a.less(b));
        }

    private:
        struct raw_tag
        {
        };
        FPSAN_HOST_DEVICE constexpr Value(storage_type s, raw_tag)
            : storage_(s)
        {
        }
        FPSAN_HOST_DEVICE static constexpr Value raw(storage_type s)
        {
            return Value(s, raw_tag{});
        }

        // float -> stored representation: the FPSan integer payload in FPSan mode,
        // the float itself otherwise. Shared by both converting constructors.
        FPSAN_HOST_DEVICE static constexpr storage_type from_float(float_type v)
        {
            if constexpr(semantics == Semantics::Triton)
                return static_cast<storage_type>(embed(v));
            else
                return static_cast<storage_type>(v);
        }

        FPSAN_HOST_DEVICE static constexpr bits_type embed(float_type v)
        {
            return detail::bits_embed<bits_type>(config, to_bits(v));
        }
        FPSAN_HOST_DEVICE static constexpr float_type unembed(bits_type p)
        {
            return from_bits(detail::bits_unembed<bits_type>(config, p));
        }
        FPSAN_HOST_DEVICE static constexpr bits_type to_bits(float_type v)
        {
            return __builtin_bit_cast(bits_type, v);
        }
        FPSAN_HOST_DEVICE static constexpr float_type from_bits(bits_type b)
        {
            return __builtin_bit_cast(float_type, b);
        }

        FPSAN_HOST_DEVICE constexpr Value combine_add(Value o) const
        {
            if constexpr(semantics == Semantics::Triton)
                return raw(
                    static_cast<storage_type>(detail::ring_add(config, storage_, o.storage_)));
            else
                return raw(static_cast<storage_type>(storage_ + o.storage_));
        }
        FPSAN_HOST_DEVICE constexpr Value combine_sub(Value o) const
        {
            if constexpr(semantics == Semantics::Triton)
                return raw(
                    static_cast<storage_type>(detail::ring_sub(config, storage_, o.storage_)));
            else
                return raw(static_cast<storage_type>(storage_ - o.storage_));
        }
        FPSAN_HOST_DEVICE constexpr Value combine_mul(Value o) const
        {
            if constexpr(semantics == Semantics::Triton)
                return raw(
                    static_cast<storage_type>(detail::ring_mul(config, storage_, o.storage_)));
            else
                return raw(static_cast<storage_type>(storage_ * o.storage_));
        }
        FPSAN_HOST_DEVICE constexpr Value combine_div(Value o) const
        {
            if constexpr(semantics == Semantics::Triton)
                return raw(
                    static_cast<storage_type>(detail::ring_div(config, storage_, o.storage_)));
            else
                return raw(static_cast<storage_type>(storage_ / o.storage_));
        }

        FPSAN_HOST_DEVICE constexpr cmp_t eq(Value o) const
        {
            return storage_ == o.storage_;
        }
        FPSAN_HOST_DEVICE constexpr cmp_t less(Value o) const
        {
            if constexpr(semantics == Semantics::Triton)
                return __builtin_bit_cast(signed_bits_type, storage_)
                       < __builtin_bit_cast(signed_bits_type, o.storage_);
            else
                return storage_ < o.storage_;
        }
        // Logical-not of a comparison result: `!` for a scalar bool, bitwise `~` for
        // a vector mask (0/-1 lanes).
        FPSAN_HOST_DEVICE static constexpr cmp_t lnot(cmp_t x)
        {
            if constexpr(is_vector)
                return ~x;
            else
                return !x;
        }

        storage_type storage_;
    };

    // ---- trait + clean error for mixing incompatible instantiations ------------
    template <class T>
    struct is_value : std::false_type
    {
    };
    template <class FT, Semantics S, Conversions C>
    struct is_value<Value<FT, S, C>> : std::true_type
    {
    };
    template <class T>
    inline constexpr bool is_value_v = is_value<T>::value;

    // A scalar that may mix with a *scalar* Value<FT, S, Conversions::Implicit>:
    // any built-in arithmetic type, plus FT itself (covers _Float16 / __bf16). Not
    // enabled for vector Values (no scalar->vector broadcast in these operators).
    template <class U, class FT>
    inline constexpr bool is_fpsan_scalar_v
        = !detail::is_clang_vector_v<FT> && (std::is_arithmetic_v<U> || std::is_same_v<U, FT>)
          && !is_value_v<U>;

#define FPSAN_DEFINE_MIXED_ARITH(OP)                                                               \
    template <class FT, Semantics S, class U, std::enable_if_t<is_fpsan_scalar_v<U, FT>, int> = 0> \
    FPSAN_HOST_DEVICE constexpr Value<FT, S, Conversions::Implicit> operator OP(                   \
        Value<FT, S, Conversions::Implicit> a, U b)                                                \
    {                                                                                              \
        return a OP Value<FT, S, Conversions::Implicit>(static_cast<FT>(b));                       \
    }                                                                                              \
    template <class FT, Semantics S, class U, std::enable_if_t<is_fpsan_scalar_v<U, FT>, int> = 0> \
    FPSAN_HOST_DEVICE constexpr Value<FT, S, Conversions::Implicit> operator OP(                   \
        U a, Value<FT, S, Conversions::Implicit> b)                                                \
    {                                                                                              \
        return Value<FT, S, Conversions::Implicit>(static_cast<FT>(a)) OP b;                       \
    }
    FPSAN_DEFINE_MIXED_ARITH(+)
    FPSAN_DEFINE_MIXED_ARITH(-)
    FPSAN_DEFINE_MIXED_ARITH(*)
    FPSAN_DEFINE_MIXED_ARITH(/)
#undef FPSAN_DEFINE_MIXED_ARITH

#define FPSAN_DEFINE_MIXED_CMP(OP)                                                                 \
    template <class FT, Semantics S, class U, std::enable_if_t<is_fpsan_scalar_v<U, FT>, int> = 0> \
    FPSAN_HOST_DEVICE constexpr bool operator OP(Value<FT, S, Conversions::Implicit> a, U b)       \
    {                                                                                              \
        return a OP Value<FT, S, Conversions::Implicit>(static_cast<FT>(b));                       \
    }                                                                                              \
    template <class FT, Semantics S, class U, std::enable_if_t<is_fpsan_scalar_v<U, FT>, int> = 0> \
    FPSAN_HOST_DEVICE constexpr bool operator OP(U a, Value<FT, S, Conversions::Implicit> b)       \
    {                                                                                              \
        return Value<FT, S, Conversions::Implicit>(static_cast<FT>(a)) OP b;                       \
    }
    FPSAN_DEFINE_MIXED_CMP(==)
    FPSAN_DEFINE_MIXED_CMP(!=)
    FPSAN_DEFINE_MIXED_CMP(<)
    FPSAN_DEFINE_MIXED_CMP(>)
    FPSAN_DEFINE_MIXED_CMP(<=)
    FPSAN_DEFINE_MIXED_CMP(>=)
#undef FPSAN_DEFINE_MIXED_CMP

#define FPSAN_DEFINE_MISMATCH_OP(OP)                                                              \
    template <class A,                                                                            \
              class B,                                                                            \
              std::enable_if_t<is_value_v<A> && is_value_v<B> && !std::is_same_v<A, B>, int> = 0> \
    FPSAN_HOST_DEVICE constexpr auto operator OP(const A&, const B&)                              \
    {                                                                                             \
        static_assert(detail::always_false<A>,                                                    \
                      "fpsan: cannot combine Value operands of different types "                  \
                      "in one operation (float_type, semantics, and conversions "                 \
                      "must all match).");                                                        \
        return false;                                                                             \
    }
    FPSAN_DEFINE_MISMATCH_OP(+)
    FPSAN_DEFINE_MISMATCH_OP(-)
    FPSAN_DEFINE_MISMATCH_OP(*)
    FPSAN_DEFINE_MISMATCH_OP(/)
    FPSAN_DEFINE_MISMATCH_OP(==)
    FPSAN_DEFINE_MISMATCH_OP(!=)
    FPSAN_DEFINE_MISMATCH_OP(<)
    FPSAN_DEFINE_MISMATCH_OP(>)
    FPSAN_DEFINE_MISMATCH_OP(<=)
    FPSAN_DEFINE_MISMATCH_OP(>=)
#undef FPSAN_DEFINE_MISMATCH_OP

} // namespace fpsan

#endif // FPSAN_VALUE_HPP
