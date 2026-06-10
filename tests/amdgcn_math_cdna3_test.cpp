// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/amdgcn_math_cdna3_test.cpp
//
// Focused gfx94x math coverage. The broader RDNA4 math test also contains
// dot4/fdot2 variants that are not exposed on CDNA3, so this file covers only
// the scalar f32 math builtins and the fdot2 builtin visible on gfx942.
#include "fpsan/amdgcn_math.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;
static constexpr int         kN  = 32;

namespace
{
    std::vector<float> make_positive_inputs()
    {
        std::vector<float> v(kN);
        std::mt19937       rng = fpsan_test::make_rng();
        for(auto& x : v)
            x = fpsan_test::pick_quarter<float>(rng, 4, 36);
        return v;
    }

    std::vector<float> make_signed_inputs()
    {
        std::vector<float> v(kN);
        std::mt19937       rng = fpsan_test::make_rng();
        for(auto& x : v)
            x = fpsan_test::pick_quarter<float>(rng, -16, 16);
        return v;
    }
} // namespace

#define CDNA3_UNARY_KERNEL(NAME, BUILTIN, FPSAN_OP)                                      \
    __global__ void k_##NAME(const float* in, float* direct, float* wrapper,             \
                             std::uint32_t* pay_direct, std::uint32_t* pay_wrapper)      \
    {                                                                                    \
        const int i = threadIdx.x;                                                       \
        const float x = in[i];                                                           \
        direct[i] = BUILTIN(x);                                                          \
        Value<float, Semantics::Float, kCC> vf{x};                                       \
        wrapper[i] = fpsan::NAME<Semantics::Float, kCC>(vf).to_float();                  \
        Value<float, Semantics::FPSan, kCC> vp{x};                                       \
        pay_direct[i] = fpsan::FPSAN_OP(vp).fpsan_payload();                             \
        pay_wrapper[i] = fpsan::NAME<Semantics::FPSan, kCC>(vp).fpsan_payload();         \
    }

CDNA3_UNARY_KERNEL(amdgcn_rcpf, __builtin_amdgcn_rcpf, rcp)
CDNA3_UNARY_KERNEL(amdgcn_sqrtf, __builtin_amdgcn_sqrtf, sqrt)
CDNA3_UNARY_KERNEL(amdgcn_rsqf, __builtin_amdgcn_rsqf, rsqrt)
CDNA3_UNARY_KERNEL(amdgcn_rsq_clampf, __builtin_amdgcn_rsq_clampf, rsqrt)
CDNA3_UNARY_KERNEL(amdgcn_sinf, __builtin_amdgcn_sinf, sin)
CDNA3_UNARY_KERNEL(amdgcn_cosf, __builtin_amdgcn_cosf, cos)
CDNA3_UNARY_KERNEL(amdgcn_logf, __builtin_amdgcn_logf, log)
CDNA3_UNARY_KERNEL(amdgcn_exp2f, __builtin_amdgcn_exp2f, exp2)
CDNA3_UNARY_KERNEL(amdgcn_fractf, __builtin_amdgcn_fractf, fract)
#undef CDNA3_UNARY_KERNEL

#define CDNA3_UNARY_TEST(NAME, INPUTS)                                                         \
    TEST(AmdgcnMathCdna3, NAME)                                                                \
    {                                                                                          \
        if(!have_device())                                                                     \
            GTEST_SKIP() << "no HIP device";                                                   \
        auto inputs = INPUTS();                                                                \
        float* dIn = to_dev(inputs);                                                           \
        float *dDirect, *dWrapper;                                                             \
        std::uint32_t *dPayDirect, *dPayWrapper;                                                \
        HIP_CHECK(hipMalloc(&dDirect, kN * sizeof(float)));                                    \
        HIP_CHECK(hipMalloc(&dWrapper, kN * sizeof(float)));                                   \
        HIP_CHECK(hipMalloc(&dPayDirect, kN * sizeof(std::uint32_t)));                         \
        HIP_CHECK(hipMalloc(&dPayWrapper, kN * sizeof(std::uint32_t)));                        \
        k_##NAME<<<1, kN>>>(dIn, dDirect, dWrapper, dPayDirect, dPayWrapper);                  \
        HIP_CHECK(hipDeviceSynchronize());                                                     \
        std::vector<float> direct(kN), wrapper(kN);                                            \
        std::vector<std::uint32_t> pay_direct(kN), pay_wrapper(kN);                            \
        HIP_CHECK(hipMemcpy(direct.data(), dDirect, kN * sizeof(float), hipMemcpyDeviceToHost)); \
        HIP_CHECK(hipMemcpy(wrapper.data(), dWrapper, kN * sizeof(float), hipMemcpyDeviceToHost)); \
        HIP_CHECK(hipMemcpy(pay_direct.data(), dPayDirect, kN * sizeof(std::uint32_t), hipMemcpyDeviceToHost)); \
        HIP_CHECK(hipMemcpy(pay_wrapper.data(), dPayWrapper, kN * sizeof(std::uint32_t), hipMemcpyDeviceToHost)); \
        for(int i = 0; i < kN; ++i)                                                           \
        {                                                                                      \
            EXPECT_EQ(bits_of(wrapper[i]), bits_of(direct[i])) << "Float lane " << i;          \
            EXPECT_EQ(pay_wrapper[i], pay_direct[i]) << "FPSan lane " << i;                   \
        }                                                                                      \
        (void)hipFree(dIn);                                                                    \
        (void)hipFree(dDirect);                                                                \
        (void)hipFree(dWrapper);                                                               \
        (void)hipFree(dPayDirect);                                                             \
        (void)hipFree(dPayWrapper);                                                            \
    }

CDNA3_UNARY_TEST(amdgcn_rcpf, make_positive_inputs)
CDNA3_UNARY_TEST(amdgcn_sqrtf, make_positive_inputs)
CDNA3_UNARY_TEST(amdgcn_rsqf, make_positive_inputs)
CDNA3_UNARY_TEST(amdgcn_rsq_clampf, make_positive_inputs)
CDNA3_UNARY_TEST(amdgcn_sinf, make_signed_inputs)
CDNA3_UNARY_TEST(amdgcn_cosf, make_signed_inputs)
CDNA3_UNARY_TEST(amdgcn_logf, make_positive_inputs)
CDNA3_UNARY_TEST(amdgcn_exp2f, make_signed_inputs)
CDNA3_UNARY_TEST(amdgcn_fractf, make_signed_inputs)
#undef CDNA3_UNARY_TEST

__global__ void k_fmed3f_cdna3(const float* a,
                               const float* b,
                               const float* c,
                               float*       direct,
                               float*       wrapper,
                               std::uint32_t* pay_direct,
                               std::uint32_t* pay_wrapper)
{
    const int i = threadIdx.x;
    direct[i] = __builtin_amdgcn_fmed3f(a[i], b[i], c[i]);
    Value<float, Semantics::Float, kCC> av{a[i]}, bv{b[i]}, cv{c[i]};
    wrapper[i] = fpsan::amdgcn_fmed3f<Semantics::Float, kCC>(av, bv, cv).to_float();
    Value<float, Semantics::FPSan, kCC> ap{a[i]}, bp{b[i]}, cp{c[i]};
    pay_direct[i] = fpsan::fmed3(ap, bp, cp).fpsan_payload();
    pay_wrapper[i] = fpsan::amdgcn_fmed3f<Semantics::FPSan, kCC>(ap, bp, cp).fpsan_payload();
}

TEST(AmdgcnMathCdna3, amdgcn_fmed3f)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    auto a = make_signed_inputs();
    auto b = make_signed_inputs();
    auto c = make_signed_inputs();
    float *dA = to_dev(a), *dB = to_dev(b), *dC = to_dev(c);
    float *dDirect, *dWrapper;
    std::uint32_t *dPayDirect, *dPayWrapper;
    HIP_CHECK(hipMalloc(&dDirect, kN * sizeof(float)));
    HIP_CHECK(hipMalloc(&dWrapper, kN * sizeof(float)));
    HIP_CHECK(hipMalloc(&dPayDirect, kN * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(&dPayWrapper, kN * sizeof(std::uint32_t)));
    k_fmed3f_cdna3<<<1, kN>>>(dA, dB, dC, dDirect, dWrapper, dPayDirect, dPayWrapper);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> direct(kN), wrapper(kN);
    std::vector<std::uint32_t> pay_direct(kN), pay_wrapper(kN);
    HIP_CHECK(hipMemcpy(direct.data(), dDirect, kN * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrapper.data(), dWrapper, kN * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(pay_direct.data(), dPayDirect, kN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(pay_wrapper.data(), dPayWrapper, kN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < kN; ++i)
    {
        EXPECT_EQ(bits_of(wrapper[i]), bits_of(direct[i])) << "Float lane " << i;
        EXPECT_EQ(pay_wrapper[i], pay_direct[i]) << "FPSan lane " << i;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dDirect);
    (void)hipFree(dWrapper);
    (void)hipFree(dPayDirect);
    (void)hipFree(dPayWrapper);
}

using v2h = _Float16 __attribute__((ext_vector_type(2)));

__global__ void k_fdot2_cdna3(const v2h* a,
                              const v2h* b,
                              const float* c,
                              float* direct,
                              float* wrapper,
                              std::uint32_t* pay_direct,
                              std::uint32_t* pay_wrapper)
{
    const int i = threadIdx.x;
    direct[i] = __builtin_amdgcn_fdot2(a[i], b[i], c[i], false);
    Value<v2h, Semantics::Float, kCC> av{a[i]}, bv{b[i]};
    Value<float, Semantics::Float, kCC> cv{c[i]};
    wrapper[i] = fpsan::amdgcn_fdot2<false, Semantics::Float, kCC>(av, bv, cv).to_float();
    Value<v2h, Semantics::FPSan, kCC> ap{a[i]}, bp{b[i]};
    Value<float, Semantics::FPSan, kCC> cp{c[i]};
    auto expanded = cp + fpsan::cast<float>(ap.get(0)) * fpsan::cast<float>(bp.get(0))
                    + fpsan::cast<float>(ap.get(1)) * fpsan::cast<float>(bp.get(1));
    pay_direct[i] = expanded.fpsan_payload();
    pay_wrapper[i] = fpsan::amdgcn_fdot2<false, Semantics::FPSan, kCC>(ap, bp, cp).fpsan_payload();
}

TEST(AmdgcnMathCdna3, amdgcn_fdot2)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<v2h> a(kN), b(kN);
    std::mt19937 rng = fpsan_test::make_rng();
    for(int i = 0; i < kN; ++i)
    {
        a[i] = v2h{static_cast<_Float16>(fpsan_test::pick_quarter<float>(rng, -8, 8)),
                   static_cast<_Float16>(fpsan_test::pick_quarter<float>(rng, -8, 8))};
        b[i] = v2h{static_cast<_Float16>(fpsan_test::pick_quarter<float>(rng, -8, 8)),
                   static_cast<_Float16>(fpsan_test::pick_quarter<float>(rng, -8, 8))};
    }
    auto c = make_signed_inputs();
    v2h *dA = to_dev(a), *dB = to_dev(b);
    float* dC = to_dev(c);
    float *dDirect, *dWrapper;
    std::uint32_t *dPayDirect, *dPayWrapper;
    HIP_CHECK(hipMalloc(&dDirect, kN * sizeof(float)));
    HIP_CHECK(hipMalloc(&dWrapper, kN * sizeof(float)));
    HIP_CHECK(hipMalloc(&dPayDirect, kN * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(&dPayWrapper, kN * sizeof(std::uint32_t)));
    k_fdot2_cdna3<<<1, kN>>>(dA, dB, dC, dDirect, dWrapper, dPayDirect, dPayWrapper);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> direct(kN), wrapper(kN);
    std::vector<std::uint32_t> pay_direct(kN), pay_wrapper(kN);
    HIP_CHECK(hipMemcpy(direct.data(), dDirect, kN * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrapper.data(), dWrapper, kN * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(pay_direct.data(), dPayDirect, kN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(pay_wrapper.data(), dPayWrapper, kN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < kN; ++i)
    {
        EXPECT_EQ(bits_of(wrapper[i]), bits_of(direct[i])) << "Float lane " << i;
        EXPECT_EQ(pay_wrapper[i], pay_direct[i]) << "FPSan lane " << i;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dDirect);
    (void)hipFree(dWrapper);
    (void)hipFree(dPayDirect);
    (void)hipFree(dPayWrapper);
}
