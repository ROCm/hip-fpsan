// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/amdgcn_math_test.cpp
//
// GPU tests for the AMD math intrinsic wrappers in fpsan/amdgcn_math.hpp.
// Two properties per wrapper:
//   - Float mode forwards to the builtin (the wrapper produces the same bits
//     as __builtin_amdgcn_{rcp,rsq,...}f directly).
//   - FPSan mode matches fpsan::{rcp,rsqrt,...} payload-for-payload (the
//     wrapper just routes to the tagged op).
#include "fpsan/amdgcn_math.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;

// Each test launches a kernel that applies the wrapper to lane-input values
// and compares against:
//   Float mode: direct builtin call (in the same kernel) -- bit-exact equality.
//   FPSan mode: fpsan::tagged_op called from FPSan (in the same kernel) --
//   payload-for-payload equality.

#define AMDGCN_MATH_UNARY_KERNEL(name, FT)                                        \
    __global__ void k_##name##_pair(const FT*      in,                            \
                                    FT*            direct,                        \
                                    FT*            via_wrapper,                   \
                                    std::uint32_t* pay_direct,                    \
                                    std::uint32_t* pay_wrapper)                   \
    {                                                                             \
        int i = threadIdx.x;                                                      \
        FT  x = in[i];                                                            \
        /* Float-mode: direct builtin vs wrapper. */                              \
        direct[i] = __builtin_##name(x);                                          \
        Value<FT, Semantics::Float, kCC> vf{x};                                   \
        via_wrapper[i] = static_cast<FT>(fpsan::name<Semantics::Float, kCC>(vf)); \
        /* FPSan-mode: tagged op vs wrapper. */                                   \
        Value<FT, Semantics::FPSan, kCC> vp{x};                                   \
        pay_direct[i]  = fpsan::FPSAN_OP_FOR_##name(vp).fpsan_payload();          \
        pay_wrapper[i] = fpsan::name<Semantics::FPSan, kCC>(vp).fpsan_payload();  \
    }

// Map each wrapper to its underlying fpsan:: tagged op.
#define FPSAN_OP_FOR_amdgcn_rcpf rcp
#define FPSAN_OP_FOR_amdgcn_sqrtf sqrt
#define FPSAN_OP_FOR_amdgcn_rsqf rsqrt
#define FPSAN_OP_FOR_amdgcn_sinf sin
#define FPSAN_OP_FOR_amdgcn_cosf cos
#define FPSAN_OP_FOR_amdgcn_logf log
#define FPSAN_OP_FOR_amdgcn_exp2f exp2
#define FPSAN_OP_FOR_amdgcn_fractf fract
#define FPSAN_OP_FOR_amdgcn_tanhf tanh

AMDGCN_MATH_UNARY_KERNEL(amdgcn_rcpf, float)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_sqrtf, float)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_rsqf, float)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_sinf, float)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_cosf, float)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_logf, float)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_exp2f, float)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_fractf, float)
// tanhf needs "tanh-insts" feature; not on RDNA4. Wrapper still defined.

namespace
{

    std::vector<float> make_inputs()
    {
        std::vector<float> v(32);
        std::mt19937       rng = fpsan_test::make_rng();
        // Positive, finite values for ops with restricted domain (log, sqrt, rsq).
        for(auto& x : v)
            x = fpsan_test::pick_quarter<float>(rng, 4, 36); // 1.0 .. 9.0
        return v;
    }

    template <class T>
    std::uint32_t bits_u32(T v)
    {
        std::uint32_t u = 0;
        std::memcpy(&u, &v, sizeof v);
        return u;
    }

} // namespace

#define MATH_TESTS(name)                                                                        \
    TEST(AmdgcnMath, name##_FloatMatchesBuiltin)                                                \
    {                                                                                           \
        int ndev = 0;                                                                           \
        if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)                                 \
            GTEST_SKIP() << "no HIP device";                                                    \
        auto           in  = make_inputs();                                                     \
        const int      N   = static_cast<int>(in.size());                                       \
        float*         dIn = to_dev(in);                                                        \
        float *        dDirect, *dWrap;                                                         \
        std::uint32_t *dPdir, *dPwrap;                                                          \
        HIP_CHECK(hipMalloc(&dDirect, N * sizeof(float)));                                      \
        HIP_CHECK(hipMalloc(&dWrap, N * sizeof(float)));                                        \
        HIP_CHECK(hipMalloc(&dPdir, N * sizeof(std::uint32_t)));                                \
        HIP_CHECK(hipMalloc(&dPwrap, N * sizeof(std::uint32_t)));                               \
        k_##name##_pair<<<1, N>>>(dIn, dDirect, dWrap, dPdir, dPwrap);                          \
        HIP_CHECK(hipDeviceSynchronize());                                                      \
        std::vector<float>         direct(N), wrap(N);                                          \
        std::vector<std::uint32_t> pdir(N), pwrap(N);                                           \
        HIP_CHECK(hipMemcpy(direct.data(), dDirect, N * sizeof(float), hipMemcpyDeviceToHost)); \
        HIP_CHECK(hipMemcpy(wrap.data(), dWrap, N * sizeof(float), hipMemcpyDeviceToHost));     \
        HIP_CHECK(                                                                              \
            hipMemcpy(pdir.data(), dPdir, N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));   \
        HIP_CHECK(                                                                              \
            hipMemcpy(pwrap.data(), dPwrap, N * sizeof(std::uint32_t), hipMemcpyDeviceToHost)); \
        for(int i = 0; i < N; ++i)                                                              \
        {                                                                                       \
            EXPECT_EQ(bits_u32(wrap[i]), bits_u32(direct[i])) << "Float lane " << i;            \
            EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;                                 \
        }                                                                                       \
        (void)hipFree(dIn);                                                                     \
        (void)hipFree(dDirect);                                                                 \
        (void)hipFree(dWrap);                                                                   \
        (void)hipFree(dPdir);                                                                   \
        (void)hipFree(dPwrap);                                                                  \
    }

MATH_TESTS(amdgcn_rcpf)
MATH_TESTS(amdgcn_sqrtf)
MATH_TESTS(amdgcn_rsqf)
MATH_TESTS(amdgcn_sinf)
MATH_TESTS(amdgcn_cosf)
MATH_TESTS(amdgcn_logf)
MATH_TESTS(amdgcn_exp2f)
MATH_TESTS(amdgcn_fractf)
// tanhf needs "tanh-insts" feature; not on RDNA4.

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
    Value<v2h, Semantics::Float, kCC>   va{ai}, vb{bi};
    Value<float, Semantics::Float, kCC> vc{ci};
    wrapper[i] = static_cast<float>(fpsan::amdgcn_fdot2<false, Semantics::Float, kCC>(va, vb, vc));
    Value<v2h, Semantics::FPSan, kCC>   vap{ai}, vbp{bi};
    Value<float, Semantics::FPSan, kCC> vcp{ci};
    auto expanded = vcp + fpsan::cast<float>(vap.get(0)) * fpsan::cast<float>(vbp.get(0))
                    + fpsan::cast<float>(vap.get(1)) * fpsan::cast<float>(vbp.get(1));
    pay_direct[i] = expanded.fpsan_payload();
    pay_wrapper[i]
        = fpsan::amdgcn_fdot2<false, Semantics::FPSan, kCC>(vap, vbp, vcp).fpsan_payload();
}

// ---- fdot2_f16_f16: v2h x v2h -> f16 ---------------------------------------
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
    Value<v2h, Semantics::Float, kCC>      va{ai}, vb{bi};
    Value<_Float16, Semantics::Float, kCC> vc{ci};
    wrapper[i]
        = static_cast<_Float16>(fpsan::amdgcn_fdot2_f16_f16<Semantics::Float, kCC>(va, vb, vc));
    Value<v2h, Semantics::FPSan, kCC>      vap{ai}, vbp{bi};
    Value<_Float16, Semantics::FPSan, kCC> vcp{ci};
    auto expanded  = vcp + vap.get(0) * vbp.get(0) + vap.get(1) * vbp.get(1);
    pay_direct[i]  = static_cast<std::uint16_t>(expanded.fpsan_payload());
    pay_wrapper[i] = static_cast<std::uint16_t>(
        fpsan::amdgcn_fdot2_f16_f16<Semantics::FPSan, kCC>(vap, vbp, vcp).fpsan_payload());
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
    Value<v2bf, Semantics::Float, kCC>  va{ai}, vb{bi};
    Value<float, Semantics::Float, kCC> vc{ci};
    wrapper[i] = static_cast<float>(
        fpsan::amdgcn_fdot2_f32_bf16<false, Semantics::Float, kCC>(va, vb, vc));
    Value<v2bf, Semantics::FPSan, kCC>  vap{ai}, vbp{bi};
    Value<float, Semantics::FPSan, kCC> vcp{ci};
    auto expanded = vcp + fpsan::cast<float>(vap.get(0)) * fpsan::cast<float>(vbp.get(0))
                    + fpsan::cast<float>(vap.get(1)) * fpsan::cast<float>(vbp.get(1));
    pay_direct[i] = expanded.fpsan_payload();
    pay_wrapper[i]
        = fpsan::amdgcn_fdot2_f32_bf16<false, Semantics::FPSan, kCC>(vap, vbp, vcp).fpsan_payload();
}

namespace
{
    constexpr int kFDot2N = 32;

    std::vector<v2h> make_v2h()
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

    std::vector<v2bf> make_v2bf()
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

    std::vector<float> make_acc_f32()
    {
        std::vector<float> v(kFDot2N);
        std::mt19937       rng = fpsan_test::make_rng();
        for(auto& x : v)
            x = fpsan_test::pick_quarter<float>(rng, -4, 4);
        return v;
    }

    std::vector<_Float16> make_acc_f16()
    {
        std::vector<_Float16> v(kFDot2N);
        std::mt19937          rng = fpsan_test::make_rng();
        for(auto& x : v)
            x = static_cast<_Float16>(fpsan_test::pick_quarter<float>(rng, -4, 4));
        return v;
    }
} // namespace

TEST(AmdgcnMath, fdot2_FloatAndFpsan)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
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

TEST(AmdgcnMath, fdot2_f16_f16_FloatAndFpsan)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
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

TEST(AmdgcnMath, fdot2_f32_bf16_FloatAndFpsan)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
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

// ============================================================================
// dot4 fp8 / bf8 family (gfx12 dot11-insts): 4-element 8-bit dot product.
// Each A and B is a v4 of fp8 (e4m3) or bf8 (e5m2) packed as 32 bits per lane.
// Float-mode: wrapper bit-exact equals direct builtin.  FPSan-mode: wrapper
// payload equals explicit ring expression `acc + sum cast<f32>(a[k]) *
// cast<f32>(b[k])`.
// ============================================================================
using v4e4 = fpsan::v4e4m3_native;
using v4e5 = fpsan::v4e5m2_native;

#define DOT4_PAIR_KERNEL(NAME, AV, BV, BUILTIN)                                                  \
    __global__ void k_##NAME##_pair(const unsigned* a,                                           \
                                    const unsigned* b,                                           \
                                    const float*    c,                                           \
                                    float*          direct,                                      \
                                    float*          wrapper,                                     \
                                    std::uint32_t*  pay_direct,                                  \
                                    std::uint32_t*  pay_wrapper)                                 \
    {                                                                                            \
        int      i  = threadIdx.x;                                                               \
        unsigned ai = a[i], bi = b[i];                                                           \
        float    ci = c[i];                                                                      \
        direct[i]   = BUILTIN(ai, bi, ci);                                                       \
        AV av = __builtin_bit_cast(AV, ai);                                                      \
        BV bv = __builtin_bit_cast(BV, bi);                                                      \
        Value<AV, Semantics::Float, kCC>    avF{av};                                             \
        Value<BV, Semantics::Float, kCC>    bvF{bv};                                             \
        Value<float, Semantics::Float, kCC> cF{ci};                                              \
        wrapper[i] = static_cast<float>(fpsan::NAME<Semantics::Float, kCC>(avF, bvF, cF));       \
        Value<AV, Semantics::FPSan, kCC>    avP{av};                                             \
        Value<BV, Semantics::FPSan, kCC>    bvP{bv};                                             \
        Value<float, Semantics::FPSan, kCC> cP{ci};                                              \
        auto expanded = cP;                                                                      \
        for(int k = 0; k < 4; ++k)                                                               \
            expanded = expanded + fpsan::cast<float>(avP.get(k)) * fpsan::cast<float>(bvP.get(k)); \
        pay_direct[i]  = expanded.fpsan_payload();                                               \
        pay_wrapper[i] = fpsan::NAME<Semantics::FPSan, kCC>(avP, bvP, cP).fpsan_payload();       \
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
        std::vector<unsigned>                      v(kDot4N);
        std::mt19937                               rng = fpsan_test::make_rng();
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

#define DOT4_TEST(NAME)                                                                          \
    TEST(AmdgcnMath, NAME##_FloatAndFpsan)                                                       \
    {                                                                                            \
        int ndev = 0;                                                                            \
        if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)                                  \
            GTEST_SKIP() << "no HIP device";                                                     \
        auto a = make_packed_u32();                                                              \
        auto b = make_packed_u32();                                                              \
        auto c = make_dot4_acc();                                                                \
        unsigned *dA = to_dev(a), *dB = to_dev(b);                                               \
        float*    dC = to_dev(c);                                                                \
        float *   dDir, *dWrap;                                                                  \
        std::uint32_t *dPdir, *dPwrap;                                                           \
        HIP_CHECK(hipMalloc(&dDir, kDot4N * sizeof(float)));                                     \
        HIP_CHECK(hipMalloc(&dWrap, kDot4N * sizeof(float)));                                    \
        HIP_CHECK(hipMalloc(&dPdir, kDot4N * sizeof(std::uint32_t)));                            \
        HIP_CHECK(hipMalloc(&dPwrap, kDot4N * sizeof(std::uint32_t)));                           \
        k_##NAME##_pair<<<1, kDot4N>>>(dA, dB, dC, dDir, dWrap, dPdir, dPwrap);                  \
        HIP_CHECK(hipDeviceSynchronize());                                                       \
        std::vector<float>         dir(kDot4N), wrap(kDot4N);                                    \
        std::vector<std::uint32_t> pdir(kDot4N), pwrap(kDot4N);                                  \
        HIP_CHECK(hipMemcpy(dir.data(), dDir, kDot4N * sizeof(float), hipMemcpyDeviceToHost));   \
        HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kDot4N * sizeof(float), hipMemcpyDeviceToHost)); \
        HIP_CHECK(hipMemcpy(                                                                     \
            pdir.data(), dPdir, kDot4N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));         \
        HIP_CHECK(hipMemcpy(                                                                     \
            pwrap.data(), dPwrap, kDot4N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));       \
        for(int i = 0; i < kDot4N; ++i)                                                          \
        {                                                                                        \
            EXPECT_EQ(bits_u32(wrap[i]), bits_u32(dir[i])) << "Float lane " << i;                \
            EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;                                  \
        }                                                                                        \
        (void)hipFree(dA);                                                                       \
        (void)hipFree(dB);                                                                       \
        (void)hipFree(dC);                                                                       \
        (void)hipFree(dDir);                                                                     \
        (void)hipFree(dWrap);                                                                    \
        (void)hipFree(dPdir);                                                                    \
        (void)hipFree(dPwrap);                                                                   \
    }

DOT4_TEST(amdgcn_dot4_f32_fp8_fp8)
DOT4_TEST(amdgcn_dot4_f32_fp8_bf8)
DOT4_TEST(amdgcn_dot4_f32_bf8_fp8)
DOT4_TEST(amdgcn_dot4_f32_bf8_bf8)
#undef DOT4_TEST
