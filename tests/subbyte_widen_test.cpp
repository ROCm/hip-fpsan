// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/subbyte_widen_test.cpp
//
// Host-level contract tests for shared fp4/fp6/bf6 widening helpers. These fail
// on the current storage-payload implementation for algebraic semantics.
#include "fpsan/detail/subbyte_widen.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "subbyte_oracle.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;

namespace
{
    template <class DstFT, int Width, Semantics S>
    void expect_widen_to(std::uint32_t code, const char* label)
    {
        using Out = Value<DstFT, S, kCC>;
        const auto got
            = fpsan::detail::subbyte_widen_to<DstFT, Width, S, kCC>(code).fpsan_payload();
        const auto want = fpsan_test::canonical_subbyte_widen_payload<DstFT, Width, S, kCC>(code);
        const auto want_value = Out::from_fpsan_payload(want);
        const auto got_value  = Out::from_fpsan_payload(got);
        EXPECT_EQ(got, want) << label << " code=0x" << std::hex << code << std::dec
                             << " S=" << int(S);
        EXPECT_EQ((got_value + got_value).fpsan_payload(),
                  (want_value + want_value).fpsan_payload())
            << label << " follow-on add code=0x" << std::hex << code << std::dec << " S=" << int(S);
    }

    template <int Width, Semantics S>
    void expect_widen_f32(std::uint32_t code, const char* label)
    {
        const auto got  = fpsan::detail::subbyte_widen<Width, S, kCC>(code).fpsan_payload();
        const auto want = fpsan_test::canonical_subbyte_widen_payload<float, Width, S, kCC>(code);
        EXPECT_EQ(got, want) << label << " code=0x" << std::hex << code << std::dec
                             << " S=" << int(S);
    }

    template <Semantics S>
    void run_fp4()
    {
        for(std::uint32_t code = 0; code < 16; ++code)
        {
            expect_widen_f32<4, S>(code, "fp4->f32 helper");
            expect_widen_to<float, 4, S>(code, "fp4->f32 helper");
            expect_widen_to<_Float16, 4, S>(code, "fp4->f16 helper");
            expect_widen_to<__bf16, 4, S>(code, "fp4->bf16 helper");
        }
    }

    template <Semantics S>
    void run_fp6()
    {
        for(std::uint32_t code = 0; code < 64; ++code)
        {
            expect_widen_f32<6, S>(code, "fp6->f32 helper");
            expect_widen_to<float, 6, S>(code, "fp6->f32 helper");
            expect_widen_to<_Float16, 6, S>(code, "fp6->f16 helper");
            expect_widen_to<__bf16, 6, S>(code, "fp6->bf16 helper");
        }
    }

    template <int Width, Semantics S>
    void expect_nonfinite_narrows_to_finite_only_codes(const char* label)
    {
        if constexpr(fpsan::detail::is_algebraic_semantics(S))
        {
            using VF          = Value<float, S, kCC>;
            const auto gotNan = fpsan_test::canonical_subbyte_narrow_code<Width>(
                VF{std::numeric_limits<float>::quiet_NaN()});
            const auto gotInf = fpsan_test::canonical_subbyte_narrow_code<Width>(
                VF{std::numeric_limits<float>::infinity()});
            const auto gotNegInf = fpsan_test::canonical_subbyte_narrow_code<Width>(
                VF{-std::numeric_limits<float>::infinity()});
            EXPECT_EQ(gotNan, 0u) << label << " NaN S=" << int(S);
            EXPECT_EQ(gotInf, fpsan_test::subbyte_positive_max_code<Width>())
                << label << " +Inf S=" << int(S);
            EXPECT_EQ(gotNegInf, fpsan_test::subbyte_positive_max_code<Width>())
                << label << " -Inf S=" << int(S);
        }
    }

    template <Semantics S>
    void run_nonfinite_narrow()
    {
        expect_nonfinite_narrows_to_finite_only_codes<4, S>("fp4 non-finite narrow");
        expect_nonfinite_narrows_to_finite_only_codes<6, S>("fp6/bf6 non-finite narrow");
    }

    template <Semantics S>
    void run_fp4_field_with_mul_casts_roundtrip()
    {
        for(std::uint32_t code = 0; code < 16; ++code)
        {
            const auto widened = fpsan_test::canonical_subbyte_widen<4, S, kCC>(code);
            const auto got     = fpsan_test::canonical_subbyte_narrow_code<4>(widened);
            const auto want    = fpsan_test::canonical_subbyte_code_from_residue<4, S>(
                fpsan_test::finite_subbyte_source_payload<4, S>(code));
            EXPECT_EQ(got, want) << "fp4 cast tower round-trip code=0x" << std::hex << code
                                 << std::dec << " S=" << int(S);
        }
    }
} // namespace

TEST(SubbyteWiden, Fp4UsesCanonicalCastPolicy)
{
    FPSAN_RUN_ALL_VARIANTS(run_fp4);
}

TEST(SubbyteWiden, Fp6UsesCanonicalCastPolicy)
{
    FPSAN_RUN_ALL_VARIANTS(run_fp6);
}

TEST(SubbyteWiden, Bf6UsesCanonicalCastPolicy)
{
    FPSAN_RUN_ALL_VARIANTS(run_fp6);
}

TEST(SubbyteNarrow, AlgebraicNonFinitesUseFiniteOnlyCodes)
{
    FPSAN_RUN_ALL_VARIANTS(run_nonfinite_narrow);
}

TEST(SubbyteRoundTrip, Fp4FieldWithMulCastsUsesCastTower)
{
    run_fp4_field_with_mul_casts_roundtrip<Semantics::FieldWithMulCasts>();
    run_fp4_field_with_mul_casts_roundtrip<Semantics::FieldWithMulCasts2>();
}
