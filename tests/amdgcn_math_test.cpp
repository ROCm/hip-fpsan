// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/amdgcn_math_test.cpp
//
// GPU tests for the AMD math intrinsic wrappers in fpsan/amdgcn_math.hpp. The
// scalar math cases (rcp/rsq/sqrt/sin/cos/log/exp2/fract) are shared across the
// gfx12 family (RDNA4 and gfx1250), gfx94x, and gfx950. Two properties per
// scalar wrapper:
//   - Float mode forwards to the builtin (the wrapper produces the same bits
//     as __builtin_amdgcn_{rcp,rsq,...}f directly).
//   - FPSan mode matches fpsan::{rcp,rsqrt,...} payload-for-payload (the
//     wrapper just routes to the tagged op).
//
// The dot-product cases (fdot2 / dot4 fp8) are gated by compile definitions
// (FPSAN_TEST_ENABLE_FDOT2 / FPSAN_TEST_ENABLE_GFX12_DOT_MATH) to the families
// whose builtins lower: the dot9-12 families exist on RDNA4 and CDNA3 but not
// on gfx1250, so they are excluded from the gfx1250 build.
#include "fpsan/amdgcn_math.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC      = Conversions::Explicit;
static constexpr int         kScalarN = 32;

#ifndef FPSAN_TEST_ENABLE_FDOT2
#define FPSAN_TEST_ENABLE_FDOT2 0
#endif

#ifndef FPSAN_TEST_ENABLE_FDOT2_EXTENDED
#define FPSAN_TEST_ENABLE_FDOT2_EXTENDED 0
#endif

#ifndef FPSAN_TEST_ENABLE_GFX12_DOT_MATH
#define FPSAN_TEST_ENABLE_GFX12_DOT_MATH 0
#endif

#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1202__) || defined(__gfx1250__)
#define FPSAN_TEST_DEVICE_IS_GFX12 1
#else
#define FPSAN_TEST_DEVICE_IS_GFX12 0
#endif

namespace
{
    template <class FT>
    std::vector<FT> make_positive_inputs()
    {
        std::vector<FT> v(kScalarN);
        std::mt19937    rng = fpsan_test::make_rng();
        for(auto& x : v)
            x = fpsan_test::pick_quarter<FT>(rng, 4, 36);
        return v;
    }

    template <class FT>
    std::vector<FT> make_signed_inputs()
    {
        std::vector<FT> v(kScalarN);
        std::mt19937    rng = fpsan_test::make_rng();
        for(auto& x : v)
            x = fpsan_test::pick_quarter<FT>(rng, -16, 16);
        return v;
    }

    std::string current_arch()
    {
        int device = 0;
        if(hipGetDevice(&device) != hipSuccess)
            return {};
        hipDeviceProp_t prop{};
        if(hipGetDeviceProperties(&prop, device) != hipSuccess)
            return {};
        return prop.gcnArchName;
    }

    bool starts_with(const std::string& value, const char* prefix)
    {
        return value.rfind(prefix, 0) == 0;
    }

    [[maybe_unused]] bool current_arch_is_gfx12()
    {
        return starts_with(current_arch(), "gfx12");
    }

    [[maybe_unused]] bool current_arch_supports_fdot2()
    {
        const auto arch = current_arch();
        return starts_with(arch, "gfx11") || starts_with(arch, "gfx12")
               || starts_with(arch, "gfx940") || starts_with(arch, "gfx941")
               || starts_with(arch, "gfx942");
    }

    [[maybe_unused]] bool current_arch_supports_fdot2_extended()
    {
        const auto arch = current_arch();
        return starts_with(arch, "gfx11") || starts_with(arch, "gfx12");
    }

    template <class T>
    std::uint32_t bits_u32(T v)
    {
        std::uint32_t u = 0;
        std::memcpy(&u, &v, sizeof v);
        return u;
    }

} // namespace

#define AMDGCN_MATH_UNARY_KERNEL(NAME, FT, BUILTIN, FPSAN_OP)                                    \
    __global__ void k_##NAME(const FT*                                              in,          \
                             FT*                                                    direct,      \
                             FT*                                                    wrapper,     \
                             typename Value<FT, Semantics::Triton, kCC>::bits_type* pay_direct,  \
                             typename Value<FT, Semantics::Triton, kCC>::bits_type* pay_wrapper) \
    {                                                                                            \
        const int i = threadIdx.x;                                                               \
        const FT  x = in[i];                                                                     \
        direct[i]   = BUILTIN(x);                                                                \
        Value<FT, Semantics::Native, kCC> vf{x};                                                 \
        wrapper[i] = fpsan::NAME<Semantics::Native, kCC>(vf).to_float();                         \
        Value<FT, Semantics::Triton, kCC> vp{x};                                                 \
        pay_direct[i]  = fpsan::FPSAN_OP(vp).fpsan_payload();                                    \
        pay_wrapper[i] = fpsan::NAME<Semantics::Triton, kCC>(vp).fpsan_payload();                \
    }

AMDGCN_MATH_UNARY_KERNEL(amdgcn_rcpf, float, __builtin_amdgcn_rcpf, rcp)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_sqrtf, float, __builtin_amdgcn_sqrtf, sqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_rsqf, float, __builtin_amdgcn_rsqf, rsqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_rsq_clampf, float, __builtin_amdgcn_rsq_clampf, rsqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_sinf, float, __builtin_amdgcn_sinf, sin)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_cosf, float, __builtin_amdgcn_cosf, cos)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_logf, float, __builtin_amdgcn_logf, log)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_exp2f, float, __builtin_amdgcn_exp2f, exp2)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_fractf, float, __builtin_amdgcn_fractf, fract)

AMDGCN_MATH_UNARY_KERNEL(amdgcn_rcp, double, __builtin_amdgcn_rcp, rcp)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_sqrt, double, __builtin_amdgcn_sqrt, sqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_rsq, double, __builtin_amdgcn_rsq, rsqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_rsq_clamp, double, __builtin_amdgcn_rsq_clamp, rsqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_fract, double, __builtin_amdgcn_fract, fract)

AMDGCN_MATH_UNARY_KERNEL(amdgcn_rcph, _Float16, __builtin_amdgcn_rcph, rcp)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_sqrth, _Float16, __builtin_amdgcn_sqrth, sqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_rsqh, _Float16, __builtin_amdgcn_rsqh, rsqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_sinh, _Float16, __builtin_amdgcn_sinh, sin)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_cosh, _Float16, __builtin_amdgcn_cosh, cos)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_fracth, _Float16, __builtin_amdgcn_fracth, fract)
#undef AMDGCN_MATH_UNARY_KERNEL

#define AMDGCN_MATH_FMED3_KERNEL(NAME, FT, BUILTIN)                                              \
    __global__ void k_##NAME(const FT*                                              a,           \
                             const FT*                                              b,           \
                             const FT*                                              c,           \
                             FT*                                                    direct,      \
                             FT*                                                    wrapper,     \
                             typename Value<FT, Semantics::Triton, kCC>::bits_type* pay_direct,  \
                             typename Value<FT, Semantics::Triton, kCC>::bits_type* pay_wrapper) \
    {                                                                                            \
        const int i = threadIdx.x;                                                               \
        direct[i]   = BUILTIN(a[i], b[i], c[i]);                                                 \
        Value<FT, Semantics::Native, kCC> af{a[i]}, bf{b[i]}, cf{c[i]};                          \
        wrapper[i] = fpsan::NAME<Semantics::Native, kCC>(af, bf, cf).to_float();                 \
        Value<FT, Semantics::Triton, kCC> ap{a[i]}, bp{b[i]}, cp{c[i]};                          \
        pay_direct[i]  = fpsan::fmed3(ap, bp, cp).fpsan_payload();                               \
        pay_wrapper[i] = fpsan::NAME<Semantics::Triton, kCC>(ap, bp, cp).fpsan_payload();        \
    }

AMDGCN_MATH_FMED3_KERNEL(amdgcn_fmed3f, float, __builtin_amdgcn_fmed3f)
AMDGCN_MATH_FMED3_KERNEL(amdgcn_fmed3h, _Float16, __builtin_amdgcn_fmed3h)
#undef AMDGCN_MATH_FMED3_KERNEL

template <class FT>
void run_unary(void (*kernel)(const FT*,
                              FT*,
                              FT*,
                              typename Value<FT, Semantics::Triton, kCC>::bits_type*,
                              typename Value<FT, Semantics::Triton, kCC>::bits_type*),
               const std::vector<FT>& inputs)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";

    using Bits = typename Value<FT, Semantics::Triton, kCC>::bits_type;
    FT*   dIn  = to_dev(inputs);
    FT *  dDirect, *dWrapper;
    Bits *dPayDirect, *dPayWrapper;
    HIP_CHECK(hipMalloc(&dDirect, kScalarN * sizeof(FT)));
    HIP_CHECK(hipMalloc(&dWrapper, kScalarN * sizeof(FT)));
    HIP_CHECK(hipMalloc(&dPayDirect, kScalarN * sizeof(Bits)));
    HIP_CHECK(hipMalloc(&dPayWrapper, kScalarN * sizeof(Bits)));

    kernel<<<1, kScalarN>>>(dIn, dDirect, dWrapper, dPayDirect, dPayWrapper);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<FT>   direct(kScalarN), wrapper(kScalarN);
    std::vector<Bits> pay_direct(kScalarN), pay_wrapper(kScalarN);
    HIP_CHECK(hipMemcpy(direct.data(), dDirect, kScalarN * sizeof(FT), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrapper.data(), dWrapper, kScalarN * sizeof(FT), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pay_direct.data(), dPayDirect, kScalarN * sizeof(Bits), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pay_wrapper.data(), dPayWrapper, kScalarN * sizeof(Bits), hipMemcpyDeviceToHost));

    for(int i = 0; i < kScalarN; ++i)
    {
        EXPECT_EQ(bits_of(wrapper[i]), bits_of(direct[i])) << "Float lane " << i;
        EXPECT_EQ(pay_wrapper[i], pay_direct[i]) << "FPSan lane " << i;
    }

    (void)hipFree(dIn);
    (void)hipFree(dDirect);
    (void)hipFree(dWrapper);
    (void)hipFree(dPayDirect);
    (void)hipFree(dPayWrapper);
}

template <class FT>
void run_fmed3(void (*kernel)(const FT*,
                              const FT*,
                              const FT*,
                              FT*,
                              FT*,
                              typename Value<FT, Semantics::Triton, kCC>::bits_type*,
                              typename Value<FT, Semantics::Triton, kCC>::bits_type*))
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";

    using Bits = typename Value<FT, Semantics::Triton, kCC>::bits_type;
    auto  a    = make_signed_inputs<FT>();
    auto  b    = make_signed_inputs<FT>();
    auto  c    = make_signed_inputs<FT>();
    FT *  dA = to_dev(a), *dB = to_dev(b), *dC = to_dev(c);
    FT *  dDirect, *dWrapper;
    Bits *dPayDirect, *dPayWrapper;
    HIP_CHECK(hipMalloc(&dDirect, kScalarN * sizeof(FT)));
    HIP_CHECK(hipMalloc(&dWrapper, kScalarN * sizeof(FT)));
    HIP_CHECK(hipMalloc(&dPayDirect, kScalarN * sizeof(Bits)));
    HIP_CHECK(hipMalloc(&dPayWrapper, kScalarN * sizeof(Bits)));

    kernel<<<1, kScalarN>>>(dA, dB, dC, dDirect, dWrapper, dPayDirect, dPayWrapper);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<FT>   direct(kScalarN), wrapper(kScalarN);
    std::vector<Bits> pay_direct(kScalarN), pay_wrapper(kScalarN);
    HIP_CHECK(hipMemcpy(direct.data(), dDirect, kScalarN * sizeof(FT), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrapper.data(), dWrapper, kScalarN * sizeof(FT), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pay_direct.data(), dPayDirect, kScalarN * sizeof(Bits), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pay_wrapper.data(), dPayWrapper, kScalarN * sizeof(Bits), hipMemcpyDeviceToHost));

    for(int i = 0; i < kScalarN; ++i)
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

#define AMDGCN_MATH_UNARY_TEST(NAME, FT, INPUTS) \
    TEST(AmdgcnMath, NAME)                       \
    {                                            \
        run_unary<FT>(k_##NAME, INPUTS<FT>());   \
    }

AMDGCN_MATH_UNARY_TEST(amdgcn_rcpf, float, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_sqrtf, float, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_rsqf, float, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_rsq_clampf, float, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_sinf, float, make_signed_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_cosf, float, make_signed_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_logf, float, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_exp2f, float, make_signed_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_fractf, float, make_signed_inputs)

AMDGCN_MATH_UNARY_TEST(amdgcn_rcp, double, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_sqrt, double, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_rsq, double, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_rsq_clamp, double, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_fract, double, make_signed_inputs)

AMDGCN_MATH_UNARY_TEST(amdgcn_rcph, _Float16, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_sqrth, _Float16, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_rsqh, _Float16, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_sinh, _Float16, make_signed_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_cosh, _Float16, make_signed_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_fracth, _Float16, make_signed_inputs)
#undef AMDGCN_MATH_UNARY_TEST

TEST(AmdgcnMath, amdgcn_fmed3f)
{
    run_fmed3<float>(k_amdgcn_fmed3f);
}

TEST(AmdgcnMath, amdgcn_fmed3h)
{
    run_fmed3<_Float16>(k_amdgcn_fmed3h);
}

// tanhf/tanhh need the tanh-insts feature and log_clampf is backend-deferred
// on the audited targets, so this shared file intentionally leaves them out.

// ============================================================================
// fdot2 family.  Two properties per wrapper:
//   - Float mode bit-exact matches the underlying builtin.
//   - FPSan mode matches the expanded `acc + cast(a0)*cast(b0) +
//     cast(a1)*cast(b1)` expression (payload-for-payload).  We compute the
//     expanded expression in the same kernel using the FPSan tagged ops, so the
//     test exercises the wrapper's contract rather than re-deriving the ring
//     math host-side.
// ============================================================================

using v2h   = _Float16 __attribute__((ext_vector_type(2)));
using v2bf  = __bf16 __attribute__((ext_vector_type(2)));
using v2i16 = short __attribute__((ext_vector_type(2)));

// ---- fdot2: v2h x v2h -> f32 -----------------------------------------------
#if FPSAN_TEST_ENABLE_FDOT2 \
    && (!defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_fdot2))
__global__ void k_fdot2_pair(const v2h*     a,
                             const v2h*     b,
                             const float*   c,
                             float*         direct,
                             float*         wrapper,
                             std::uint32_t* pay_direct,
                             std::uint32_t* pay_wrapper)
{
    int   i  = threadIdx.x;
    v2h   ai = a[i], bi = b[i];
    float ci  = c[i];
    direct[i] = __builtin_amdgcn_fdot2(ai, bi, ci, false);
    Value<v2h, Semantics::Native, kCC>   va{ai}, vb{bi};
    Value<float, Semantics::Native, kCC> vc{ci};
    wrapper[i] = static_cast<float>(fpsan::amdgcn_fdot2<false, Semantics::Native, kCC>(va, vb, vc));
    Value<v2h, Semantics::Triton, kCC>   vap{ai}, vbp{bi};
    Value<float, Semantics::Triton, kCC> vcp{ci};
    auto expanded = vcp + fpsan::cast<float>(vap.get(0)) * fpsan::cast<float>(vbp.get(0))
                    + fpsan::cast<float>(vap.get(1)) * fpsan::cast<float>(vbp.get(1));
    pay_direct[i] = expanded.fpsan_payload();
    pay_wrapper[i]
        = fpsan::amdgcn_fdot2<false, Semantics::Triton, kCC>(vap, vbp, vcp).fpsan_payload();
}
#endif

// ---- fdot2_f16_f16: v2h x v2h -> f16 ---------------------------------------
#if FPSAN_TEST_ENABLE_FDOT2_EXTENDED                           \
    && (!defined(__HIP_DEVICE_COMPILE__)                       \
        || (__has_builtin(__builtin_amdgcn_fdot2_f16_f16)      \
            && __has_builtin(__builtin_amdgcn_fdot2_bf16_bf16) \
            && __has_builtin(__builtin_amdgcn_fdot2_f32_bf16)))
__global__ void k_fdot2_f16_f16_pair(const v2h*      a,
                                     const v2h*      b,
                                     const _Float16* c,
                                     _Float16*       direct,
                                     _Float16*       wrapper,
                                     std::uint16_t*  pay_direct,
                                     std::uint16_t*  pay_wrapper)
{
    int      i  = threadIdx.x;
    v2h      ai = a[i], bi = b[i];
    _Float16 ci = c[i];
    direct[i]   = __builtin_amdgcn_fdot2_f16_f16(ai, bi, ci);
    Value<v2h, Semantics::Native, kCC>      va{ai}, vb{bi};
    Value<_Float16, Semantics::Native, kCC> vc{ci};
    wrapper[i]
        = static_cast<_Float16>(fpsan::amdgcn_fdot2_f16_f16<Semantics::Native, kCC>(va, vb, vc));
    Value<v2h, Semantics::Triton, kCC>      vap{ai}, vbp{bi};
    Value<_Float16, Semantics::Triton, kCC> vcp{ci};
    auto expanded  = vcp + vap.get(0) * vbp.get(0) + vap.get(1) * vbp.get(1);
    pay_direct[i]  = static_cast<std::uint16_t>(expanded.fpsan_payload());
    pay_wrapper[i] = static_cast<std::uint16_t>(
        fpsan::amdgcn_fdot2_f16_f16<Semantics::Triton, kCC>(vap, vbp, vcp).fpsan_payload());
}

// ---- fdot2_bf16_bf16: v2bf x v2bf -> bf16 ----------------------------------
__global__ void k_fdot2_bf16_bf16_pair(const v2bf*    a,
                                       const v2bf*    b,
                                       const __bf16*  c,
                                       __bf16*        direct,
                                       __bf16*        wrapper,
                                       std::uint16_t* pay_direct,
                                       std::uint16_t* pay_wrapper)
{
    int    i  = threadIdx.x;
    v2bf   ai = a[i], bi = b[i];
    __bf16 ci  = c[i];
    v2i16  a_i = __builtin_bit_cast(v2i16, ai);
    v2i16  b_i = __builtin_bit_cast(v2i16, bi);
    short  c_i = __builtin_bit_cast(short, ci);
    short  d_i = __builtin_amdgcn_fdot2_bf16_bf16(a_i, b_i, c_i);
    direct[i]  = __builtin_bit_cast(__bf16, d_i);
    Value<v2bf, Semantics::Native, kCC>   va{ai}, vb{bi};
    Value<__bf16, Semantics::Native, kCC> vc{ci};
    wrapper[i]
        = static_cast<__bf16>(fpsan::amdgcn_fdot2_bf16_bf16<Semantics::Native, kCC>(va, vb, vc));
    Value<v2bf, Semantics::Triton, kCC>   vap{ai}, vbp{bi};
    Value<__bf16, Semantics::Triton, kCC> vcp{ci};
    auto expanded  = vcp + vap.get(0) * vbp.get(0) + vap.get(1) * vbp.get(1);
    pay_direct[i]  = static_cast<std::uint16_t>(expanded.fpsan_payload());
    pay_wrapper[i] = static_cast<std::uint16_t>(
        fpsan::amdgcn_fdot2_bf16_bf16<Semantics::Triton, kCC>(vap, vbp, vcp).fpsan_payload());
}

// ---- fdot2_f32_bf16: v2bf x v2bf -> f32 ------------------------------------
__global__ void k_fdot2_f32_bf16_pair(const v2bf*    a,
                                      const v2bf*    b,
                                      const float*   c,
                                      float*         direct,
                                      float*         wrapper,
                                      std::uint32_t* pay_direct,
                                      std::uint32_t* pay_wrapper)
{
    int   i  = threadIdx.x;
    v2bf  ai = a[i], bi = b[i];
    float ci  = c[i];
    v2i16 a_i = __builtin_bit_cast(v2i16, ai);
    v2i16 b_i = __builtin_bit_cast(v2i16, bi);
    direct[i] = __builtin_amdgcn_fdot2_f32_bf16(a_i, b_i, ci, false);
    Value<v2bf, Semantics::Native, kCC>  va{ai}, vb{bi};
    Value<float, Semantics::Native, kCC> vc{ci};
    wrapper[i] = static_cast<float>(
        fpsan::amdgcn_fdot2_f32_bf16<false, Semantics::Native, kCC>(va, vb, vc));
    Value<v2bf, Semantics::Triton, kCC>  vap{ai}, vbp{bi};
    Value<float, Semantics::Triton, kCC> vcp{ci};
    auto expanded = vcp + fpsan::cast<float>(vap.get(0)) * fpsan::cast<float>(vbp.get(0))
                    + fpsan::cast<float>(vap.get(1)) * fpsan::cast<float>(vbp.get(1));
    pay_direct[i]  = expanded.fpsan_payload();
    pay_wrapper[i] = fpsan::amdgcn_fdot2_f32_bf16<false, Semantics::Triton, kCC>(vap, vbp, vcp)
                         .fpsan_payload();
}
#endif

namespace
{
    constexpr int kFDot2N = 32;

    [[maybe_unused]] std::vector<v2h> make_v2h()
    {
        std::vector<v2h> v(kFDot2N);
        std::mt19937     rng = fpsan_test::make_rng();
        for(auto& x : v)
        {
            float a = fpsan_test::pick_quarter<float>(rng, -8, 8);
            float b = fpsan_test::pick_quarter<float>(rng, -8, 8);
            x       = v2h{static_cast<_Float16>(a), static_cast<_Float16>(b)};
        }
        return v;
    }

    [[maybe_unused]] std::vector<v2bf> make_v2bf()
    {
        std::vector<v2bf> v(kFDot2N);
        std::mt19937      rng = fpsan_test::make_rng();
        for(auto& x : v)
        {
            float a = fpsan_test::pick_quarter<float>(rng, -8, 8);
            float b = fpsan_test::pick_quarter<float>(rng, -8, 8);
            x       = v2bf{static_cast<__bf16>(a), static_cast<__bf16>(b)};
        }
        return v;
    }

    [[maybe_unused]] std::vector<float> make_acc_f32()
    {
        std::vector<float> v(kFDot2N);
        std::mt19937       rng = fpsan_test::make_rng();
        for(auto& x : v)
            x = fpsan_test::pick_quarter<float>(rng, -4, 4);
        return v;
    }

    [[maybe_unused]] std::vector<_Float16> make_acc_f16()
    {
        std::vector<_Float16> v(kFDot2N);
        std::mt19937          rng = fpsan_test::make_rng();
        for(auto& x : v)
            x = static_cast<_Float16>(fpsan_test::pick_quarter<float>(rng, -4, 4));
        return v;
    }
} // namespace

#if FPSAN_TEST_ENABLE_FDOT2 \
    && (!defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_fdot2))
TEST(AmdgcnMath, fdot2_FloatAndFpsan)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    if(!current_arch_supports_fdot2())
        GTEST_SKIP() << "current device does not support fdot2";
    auto           a = make_v2h(), b = make_v2h();
    auto           c  = make_acc_f32();
    v2h *          dA = to_dev(a), *dB = to_dev(b);
    float*         dC = to_dev(c);
    float *        dDir, *dWrap;
    std::uint32_t *dPdir, *dPwrap;
    HIP_CHECK(hipMalloc(&dDir, kFDot2N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dWrap, kFDot2N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dPdir, kFDot2N * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(&dPwrap, kFDot2N * sizeof(std::uint32_t)));
    k_fdot2_pair<<<1, kFDot2N>>>(dA, dB, dC, dDir, dWrap, dPdir, dPwrap);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float>         dir(kFDot2N), wrap(kFDot2N);
    std::vector<std::uint32_t> pdir(kFDot2N), pwrap(kFDot2N);
    HIP_CHECK(hipMemcpy(dir.data(), dDir, kFDot2N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kFDot2N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pdir.data(), dPdir, kFDot2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pwrap.data(), dPwrap, kFDot2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < kFDot2N; ++i)
    {
        EXPECT_EQ(bits_u32(wrap[i]), bits_u32(dir[i])) << "Float lane " << i;
        EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dDir);
    (void)hipFree(dWrap);
    (void)hipFree(dPdir);
    (void)hipFree(dPwrap);
}
#endif

#if FPSAN_TEST_ENABLE_FDOT2_EXTENDED                           \
    && (!defined(__HIP_DEVICE_COMPILE__)                       \
        || (__has_builtin(__builtin_amdgcn_fdot2_f16_f16)      \
            && __has_builtin(__builtin_amdgcn_fdot2_bf16_bf16) \
            && __has_builtin(__builtin_amdgcn_fdot2_f32_bf16)))
TEST(AmdgcnMath, fdot2_f16_f16_FloatAndFpsan)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    if(!current_arch_supports_fdot2_extended())
        GTEST_SKIP() << "current device does not support extended fdot2";
    auto           a = make_v2h(), b = make_v2h();
    auto           c  = make_acc_f16();
    v2h *          dA = to_dev(a), *dB = to_dev(b);
    _Float16*      dC = to_dev(c);
    _Float16 *     dDir, *dWrap;
    std::uint16_t *dPdir, *dPwrap;
    HIP_CHECK(hipMalloc(&dDir, kFDot2N * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&dWrap, kFDot2N * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&dPdir, kFDot2N * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&dPwrap, kFDot2N * sizeof(std::uint16_t)));
    k_fdot2_f16_f16_pair<<<1, kFDot2N>>>(dA, dB, dC, dDir, dWrap, dPdir, dPwrap);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<_Float16>      dir(kFDot2N), wrap(kFDot2N);
    std::vector<std::uint16_t> pdir(kFDot2N), pwrap(kFDot2N);
    HIP_CHECK(hipMemcpy(dir.data(), dDir, kFDot2N * sizeof(_Float16), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kFDot2N * sizeof(_Float16), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pdir.data(), dPdir, kFDot2N * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pwrap.data(), dPwrap, kFDot2N * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < kFDot2N; ++i)
    {
        std::uint16_t bw = 0, bd = 0;
        std::memcpy(&bw, &wrap[i], sizeof bw);
        std::memcpy(&bd, &dir[i], sizeof bd);
        EXPECT_EQ(bw, bd) << "Float lane " << i;
        EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dDir);
    (void)hipFree(dWrap);
    (void)hipFree(dPdir);
    (void)hipFree(dPwrap);
}

TEST(AmdgcnMath, fdot2_bf16_bf16_FloatAndFpsan)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    if(!current_arch_supports_fdot2_extended())
        GTEST_SKIP() << "current device does not support extended fdot2";
    auto                a = make_v2bf(), b = make_v2bf();
    auto                cf = make_acc_f32();
    std::vector<__bf16> c(kFDot2N);
    for(int i = 0; i < kFDot2N; ++i)
        c[i] = static_cast<__bf16>(cf[i]);
    v2bf *         dA = to_dev(a), *dB = to_dev(b);
    __bf16*        dC = to_dev(c);
    __bf16 *       dDir, *dWrap;
    std::uint16_t *dPdir, *dPwrap;
    HIP_CHECK(hipMalloc(&dDir, kFDot2N * sizeof(__bf16)));
    HIP_CHECK(hipMalloc(&dWrap, kFDot2N * sizeof(__bf16)));
    HIP_CHECK(hipMalloc(&dPdir, kFDot2N * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&dPwrap, kFDot2N * sizeof(std::uint16_t)));
    k_fdot2_bf16_bf16_pair<<<1, kFDot2N>>>(dA, dB, dC, dDir, dWrap, dPdir, dPwrap);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<__bf16>        dir(kFDot2N), wrap(kFDot2N);
    std::vector<std::uint16_t> pdir(kFDot2N), pwrap(kFDot2N);
    HIP_CHECK(hipMemcpy(dir.data(), dDir, kFDot2N * sizeof(__bf16), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kFDot2N * sizeof(__bf16), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pdir.data(), dPdir, kFDot2N * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pwrap.data(), dPwrap, kFDot2N * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < kFDot2N; ++i)
    {
        std::uint16_t bw = 0, bd = 0;
        std::memcpy(&bw, &wrap[i], sizeof bw);
        std::memcpy(&bd, &dir[i], sizeof bd);
        EXPECT_EQ(bw, bd) << "Float lane " << i;
        EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dDir);
    (void)hipFree(dWrap);
    (void)hipFree(dPdir);
    (void)hipFree(dPwrap);
}

TEST(AmdgcnMath, fdot2_f32_bf16_FloatAndFpsan)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    if(!current_arch_supports_fdot2_extended())
        GTEST_SKIP() << "current device does not support extended fdot2";
    auto           a = make_v2bf(), b = make_v2bf();
    auto           c  = make_acc_f32();
    v2bf *         dA = to_dev(a), *dB = to_dev(b);
    float*         dC = to_dev(c);
    float *        dDir, *dWrap;
    std::uint32_t *dPdir, *dPwrap;
    HIP_CHECK(hipMalloc(&dDir, kFDot2N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dWrap, kFDot2N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dPdir, kFDot2N * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(&dPwrap, kFDot2N * sizeof(std::uint32_t)));
    k_fdot2_f32_bf16_pair<<<1, kFDot2N>>>(dA, dB, dC, dDir, dWrap, dPdir, dPwrap);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float>         dir(kFDot2N), wrap(kFDot2N);
    std::vector<std::uint32_t> pdir(kFDot2N), pwrap(kFDot2N);
    HIP_CHECK(hipMemcpy(dir.data(), dDir, kFDot2N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kFDot2N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pdir.data(), dPdir, kFDot2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pwrap.data(), dPwrap, kFDot2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < kFDot2N; ++i)
    {
        EXPECT_EQ(bits_u32(wrap[i]), bits_u32(dir[i])) << "Float lane " << i;
        EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dDir);
    (void)hipFree(dWrap);
    (void)hipFree(dPdir);
    (void)hipFree(dPwrap);
}
#endif

// ============================================================================
// dot4 fp8 / bf8 family (gfx12 dot11-insts): 4-element 8-bit dot product.
// Each A and B is a v4 of fp8 (e4m3) or bf8 (e5m2) packed as 32 bits per lane.
// Float-mode: wrapper bit-exact equals direct builtin.  FPSan-mode: wrapper
// payload equals explicit ring expression `acc + sum cast<f32>(a[k]) *
// cast<f32>(b[k])`.
// ============================================================================
#if FPSAN_TEST_ENABLE_GFX12_DOT_MATH \
    && (!defined(__HIP_DEVICE_COMPILE__) || FPSAN_TEST_DEVICE_IS_GFX12)
using v4e4 = fpsan::v4e4m3_native;
using v4e5 = fpsan::v4e5m2_native;

#define DOT4_PAIR_KERNEL(NAME, AV, BV, BUILTIN)                                                    \
    __global__ void k_##NAME##_pair(const unsigned* a,                                             \
                                    const unsigned* b,                                             \
                                    const float*    c,                                             \
                                    float*          direct,                                        \
                                    float*          wrapper,                                       \
                                    std::uint32_t*  pay_direct,                                    \
                                    std::uint32_t*  pay_wrapper)                                   \
    {                                                                                              \
        int      i  = threadIdx.x;                                                                 \
        unsigned ai = a[i], bi = b[i];                                                             \
        float    ci                             = c[i];                                            \
        direct[i]                               = BUILTIN(ai, bi, ci);                             \
        AV                                   av = __builtin_bit_cast(AV, ai);                      \
        BV                                   bv = __builtin_bit_cast(BV, bi);                      \
        Value<AV, Semantics::Native, kCC>    avF{av};                                              \
        Value<BV, Semantics::Native, kCC>    bvF{bv};                                              \
        Value<float, Semantics::Native, kCC> cF{ci};                                               \
        wrapper[i] = static_cast<float>(fpsan::NAME<Semantics::Native, kCC>(avF, bvF, cF));        \
        Value<AV, Semantics::Triton, kCC>    avP{av};                                              \
        Value<BV, Semantics::Triton, kCC>    bvP{bv};                                              \
        Value<float, Semantics::Triton, kCC> cP{ci};                                               \
        auto                                 expanded = cP;                                        \
        for(int k = 0; k < 4; ++k)                                                                 \
            expanded = expanded + fpsan::cast<float>(avP.get(k)) * fpsan::cast<float>(bvP.get(k)); \
        pay_direct[i]  = expanded.fpsan_payload();                                                 \
        pay_wrapper[i] = fpsan::NAME<Semantics::Triton, kCC>(avP, bvP, cP).fpsan_payload();        \
    }

DOT4_PAIR_KERNEL(amdgcn_dot4_f32_fp8_fp8, v4e4, v4e4, __builtin_amdgcn_dot4_f32_fp8_fp8)
DOT4_PAIR_KERNEL(amdgcn_dot4_f32_fp8_bf8, v4e4, v4e5, __builtin_amdgcn_dot4_f32_fp8_bf8)
DOT4_PAIR_KERNEL(amdgcn_dot4_f32_bf8_fp8, v4e5, v4e4, __builtin_amdgcn_dot4_f32_bf8_fp8)
DOT4_PAIR_KERNEL(amdgcn_dot4_f32_bf8_bf8, v4e5, v4e5, __builtin_amdgcn_dot4_f32_bf8_bf8)
#undef DOT4_PAIR_KERNEL

namespace
{
    constexpr int kDot4N = 32;

    std::vector<unsigned> make_packed_u32()
    {
        std::vector<unsigned>                        v(kDot4N);
        std::mt19937                                 rng = fpsan_test::make_rng();
        std::uniform_int_distribution<std::uint32_t> dist;
        for(auto& x : v)
            x = dist(rng);
        return v;
    }

    std::vector<float> make_dot4_acc()
    {
        std::vector<float> v(kDot4N);
        std::mt19937       rng = fpsan_test::make_rng();
        for(auto& x : v)
            x = fpsan_test::pick_quarter<float>(rng, -4, 4);
        return v;
    }
} // namespace

#define DOT4_TEST(NAME)                                                                            \
    TEST(AmdgcnMath, NAME##_FloatAndFpsan)                                                         \
    {                                                                                              \
        if(!have_device())                                                                         \
            GTEST_SKIP() << "no HIP device";                                                       \
        if(!current_arch_is_gfx12())                                                               \
            GTEST_SKIP() << "current device is not gfx12";                                         \
        auto           a  = make_packed_u32();                                                     \
        auto           b  = make_packed_u32();                                                     \
        auto           c  = make_dot4_acc();                                                       \
        unsigned *     dA = to_dev(a), *dB = to_dev(b);                                            \
        float*         dC = to_dev(c);                                                             \
        float *        dDir, *dWrap;                                                               \
        std::uint32_t *dPdir, *dPwrap;                                                             \
        HIP_CHECK(hipMalloc(&dDir, kDot4N * sizeof(float)));                                       \
        HIP_CHECK(hipMalloc(&dWrap, kDot4N * sizeof(float)));                                      \
        HIP_CHECK(hipMalloc(&dPdir, kDot4N * sizeof(std::uint32_t)));                              \
        HIP_CHECK(hipMalloc(&dPwrap, kDot4N * sizeof(std::uint32_t)));                             \
        k_##NAME##_pair<<<1, kDot4N>>>(dA, dB, dC, dDir, dWrap, dPdir, dPwrap);                    \
        HIP_CHECK(hipDeviceSynchronize());                                                         \
        std::vector<float>         dir(kDot4N), wrap(kDot4N);                                      \
        std::vector<std::uint32_t> pdir(kDot4N), pwrap(kDot4N);                                    \
        HIP_CHECK(hipMemcpy(dir.data(), dDir, kDot4N * sizeof(float), hipMemcpyDeviceToHost));     \
        HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kDot4N * sizeof(float), hipMemcpyDeviceToHost));   \
        HIP_CHECK(                                                                                 \
            hipMemcpy(pdir.data(), dPdir, kDot4N * sizeof(std::uint32_t), hipMemcpyDeviceToHost)); \
        HIP_CHECK(hipMemcpy(                                                                       \
            pwrap.data(), dPwrap, kDot4N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));         \
        for(int i = 0; i < kDot4N; ++i)                                                            \
        {                                                                                          \
            EXPECT_EQ(bits_u32(wrap[i]), bits_u32(dir[i])) << "Float lane " << i;                  \
            EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;                                    \
        }                                                                                          \
        (void)hipFree(dA);                                                                         \
        (void)hipFree(dB);                                                                         \
        (void)hipFree(dC);                                                                         \
        (void)hipFree(dDir);                                                                       \
        (void)hipFree(dWrap);                                                                      \
        (void)hipFree(dPdir);                                                                      \
        (void)hipFree(dPwrap);                                                                     \
    }

DOT4_TEST(amdgcn_dot4_f32_fp8_fp8)
DOT4_TEST(amdgcn_dot4_f32_fp8_bf8)
DOT4_TEST(amdgcn_dot4_f32_bf8_fp8)
DOT4_TEST(amdgcn_dot4_f32_bf8_bf8)
#undef DOT4_TEST
#endif
