// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_test.cpp
//
// GPU tests for fpsan/amdgcn_cvt.hpp.
#include "fpsan/amdgcn_cvt.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;

#ifndef FPSAN_TEST_ENABLE_FP8_CVT
#define FPSAN_TEST_ENABLE_FP8_CVT 0
#endif
#ifndef FPSAN_TEST_ENABLE_PKRTZ_CVT
#define FPSAN_TEST_ENABLE_PKRTZ_CVT 1
#endif

#if FPSAN_TEST_ENABLE_PKRTZ_CVT
__global__ void k_cvt_pkrtz_float(const float* a, const float* b, std::uint32_t* out)
{
    int                                  i = threadIdx.x;
    Value<float, Semantics::Native, kCC> av{a[i]}, bv{b[i]};
    auto r   = fpsan::amdgcn_cvt_pkrtz<Semantics::Native, kCC>(av, bv);
    auto vec = static_cast<fpsan::v2h_native>(r);
    out[i]   = __builtin_bit_cast(std::uint32_t, vec);
}

__global__ void k_cvt_pkrtz_fpsan(const float* a, const float* b, std::uint32_t* out)
{
    int                                  i = threadIdx.x;
    Value<float, Semantics::Triton, kCC> av{a[i]}, bv{b[i]};
    auto r   = fpsan::amdgcn_cvt_pkrtz<Semantics::Triton, kCC>(av, bv);
    auto pay = r.fpsan_payload(); // v2u16
    out[i]   = __builtin_bit_cast(std::uint32_t, pay);
}

TEST(Cvt, PkrtzFloatMatchesBuiltin)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    std::vector<float> a(32), b(32);
    std::mt19937       rng = fpsan_test::make_rng();
    for(int i = 0; i < 32; ++i)
    {
        a[i] = fpsan_test::pick_quarter<float>(rng, -40, 40);
        b[i] = fpsan_test::pick_quarter<float>(rng, -40, 40);
    }
    float *        dA, *dB;
    std::uint32_t* dOut;
    HIP_CHECK(hipMalloc(&dA, 32 * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, 32 * sizeof(float)));
    HIP_CHECK(hipMalloc(&dOut, 32 * sizeof(std::uint32_t)));
    HIP_CHECK(hipMemcpy(dA, a.data(), 32 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, b.data(), 32 * sizeof(float), hipMemcpyHostToDevice));
    k_cvt_pkrtz_float<<<1, 32>>>(dA, dB, dOut);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(32);
    HIP_CHECK(hipMemcpy(got.data(), dOut, 32 * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    // Float-mode result should equal pack(static_cast<_Float16>(a),
    // static_cast<_Float16>(b)) which is what cvt_pkrtz does up to RTZ rounding;
    // since our inputs are quarters in [-10,10], they are representable in
    // _Float16 exactly, so RTZ collapses to identity and the result IS the
    // pack of f16 casts.
    for(int i = 0; i < 32; ++i)
    {
        _Float16      ah = static_cast<_Float16>(a[i]);
        _Float16      bh = static_cast<_Float16>(b[i]);
        std::uint16_t ab, bb;
        std::memcpy(&ab, &ah, sizeof ab);
        std::memcpy(&bb, &bh, sizeof bb);
        std::uint32_t expected
            = static_cast<std::uint32_t>(ab) | (static_cast<std::uint32_t>(bb) << 16);
        EXPECT_EQ(got[i], expected) << "lane " << i;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dOut);
}

TEST(Cvt, PkrtzFpsanMatchesPerLaneCast)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    std::vector<float> a(32), b(32);
    std::mt19937       rng = fpsan_test::make_rng();
    for(int i = 0; i < 32; ++i)
    {
        a[i] = fpsan_test::pick_quarter<float>(rng, -40, 40);
        b[i] = fpsan_test::pick_quarter<float>(rng, -40, 40);
    }
    float *        dA, *dB;
    std::uint32_t* dOut;
    HIP_CHECK(hipMalloc(&dA, 32 * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, 32 * sizeof(float)));
    HIP_CHECK(hipMalloc(&dOut, 32 * sizeof(std::uint32_t)));
    HIP_CHECK(hipMemcpy(dA, a.data(), 32 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, b.data(), 32 * sizeof(float), hipMemcpyHostToDevice));
    k_cvt_pkrtz_fpsan<<<1, 32>>>(dA, dB, dOut);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(32);
    HIP_CHECK(hipMemcpy(got.data(), dOut, 32 * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    using F = Value<float, Semantics::Triton, kCC>;
    using H = Value<_Float16, Semantics::Triton, kCC>;
    for(int i = 0; i < 32; ++i)
    {
        H             ah       = fpsan::cast<_Float16>(F{a[i]});
        H             bh       = fpsan::cast<_Float16>(F{b[i]});
        std::uint32_t expected = static_cast<std::uint32_t>(ah.fpsan_payload())
                                 | (static_cast<std::uint32_t>(bh.fpsan_payload()) << 16);
        EXPECT_EQ(got[i], expected) << "lane " << i;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dOut);
}
#endif // FPSAN_TEST_ENABLE_PKRTZ_CVT

// ============================================================================
// FP8 conversion intrinsic wrappers.
// ============================================================================
#if FPSAN_TEST_ENABLE_FP8_CVT

namespace
{
    constexpr int kFp8N = 32;

    std::vector<int> make_packed_ints()
    {
        std::vector<int>                             v(kFp8N);
        std::mt19937                                 rng = fpsan_test::make_rng();
        std::uniform_int_distribution<std::uint32_t> dist;
        for(auto& x : v)
            x = static_cast<int>(dist(rng));
        return v;
    }

    std::vector<float> make_fp8_floats()
    {
        std::vector<float> v(kFp8N);
        std::mt19937       rng = fpsan_test::make_rng();
        // Stay inside e4m3's representable range so the f32 -> fp8 narrowing is
        // deterministic across both formats.
        for(auto& x : v)
            x = fpsan_test::pick_quarter<float>(rng, -100, 100);
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

// ---- cvt_f32_fp8 / cvt_f32_bf8 (unpack) ------------------------------------
template <int Idx>
__global__ void k_cvt_f32_fp8_pair(const int*     packed,
                                   float*         direct,
                                   float*         wrapper,
                                   std::uint32_t* pay_direct,
                                   std::uint32_t* pay_wrapper)
{
    int i     = threadIdx.x;
    direct[i] = __builtin_amdgcn_cvt_f32_fp8(packed[i], Idx);
    wrapper[i]
        = static_cast<float>(fpsan::amdgcn_cvt_f32_fp8<Idx, Semantics::Native, kCC>(packed[i]));
    const std::uint32_t u    = static_cast<std::uint32_t>(packed[i]);
    const std::uint8_t  byte = static_cast<std::uint8_t>((u >> (Idx * 8)) & 0xFFu);
    auto v        = Value<fpsan::fp8_e4m3, Semantics::Triton, kCC>::from_fpsan_payload(byte);
    pay_direct[i] = fpsan::cast<float>(v).fpsan_payload();
    pay_wrapper[i]
        = fpsan::amdgcn_cvt_f32_fp8<Idx, Semantics::Triton, kCC>(packed[i]).fpsan_payload();
}

template <int Idx>
__global__ void k_cvt_f32_bf8_pair(const int*     packed,
                                   float*         direct,
                                   float*         wrapper,
                                   std::uint32_t* pay_direct,
                                   std::uint32_t* pay_wrapper)
{
    int i     = threadIdx.x;
    direct[i] = __builtin_amdgcn_cvt_f32_bf8(packed[i], Idx);
    wrapper[i]
        = static_cast<float>(fpsan::amdgcn_cvt_f32_bf8<Idx, Semantics::Native, kCC>(packed[i]));
    const std::uint32_t u    = static_cast<std::uint32_t>(packed[i]);
    const std::uint8_t  byte = static_cast<std::uint8_t>((u >> (Idx * 8)) & 0xFFu);
    auto v        = Value<fpsan::fp8_e5m2, Semantics::Triton, kCC>::from_fpsan_payload(byte);
    pay_direct[i] = fpsan::cast<float>(v).fpsan_payload();
    pay_wrapper[i]
        = fpsan::amdgcn_cvt_f32_bf8<Idx, Semantics::Triton, kCC>(packed[i]).fpsan_payload();
}

#define CVT_F32_FP8_TEST(FAMILY, IDX)                                                             \
    TEST(Cvt, FAMILY##_Byte##IDX)                                                                 \
    {                                                                                             \
        int ndev = 0;                                                                             \
        if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)                                   \
            GTEST_SKIP() << "no HIP device";                                                      \
        auto           in = make_packed_ints();                                                   \
        int*           dIn;                                                                       \
        float *        dDir, *dWrap;                                                              \
        std::uint32_t *dPdir, *dPwrap;                                                            \
        HIP_CHECK(hipMalloc(&dIn, kFp8N * sizeof(int)));                                          \
        HIP_CHECK(hipMalloc(&dDir, kFp8N * sizeof(float)));                                       \
        HIP_CHECK(hipMalloc(&dWrap, kFp8N * sizeof(float)));                                      \
        HIP_CHECK(hipMalloc(&dPdir, kFp8N * sizeof(std::uint32_t)));                              \
        HIP_CHECK(hipMalloc(&dPwrap, kFp8N * sizeof(std::uint32_t)));                             \
        HIP_CHECK(hipMemcpy(dIn, in.data(), kFp8N * sizeof(int), hipMemcpyHostToDevice));         \
        k_##FAMILY##_pair<IDX><<<1, kFp8N>>>(dIn, dDir, dWrap, dPdir, dPwrap);                    \
        HIP_CHECK(hipDeviceSynchronize());                                                        \
        std::vector<float>         dir(kFp8N), wrap(kFp8N);                                       \
        std::vector<std::uint32_t> pdir(kFp8N), pwrap(kFp8N);                                     \
        HIP_CHECK(hipMemcpy(dir.data(), dDir, kFp8N * sizeof(float), hipMemcpyDeviceToHost));     \
        HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kFp8N * sizeof(float), hipMemcpyDeviceToHost));   \
        HIP_CHECK(                                                                                \
            hipMemcpy(pdir.data(), dPdir, kFp8N * sizeof(std::uint32_t), hipMemcpyDeviceToHost)); \
        HIP_CHECK(hipMemcpy(                                                                      \
            pwrap.data(), dPwrap, kFp8N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));         \
        for(int i = 0; i < kFp8N; ++i)                                                            \
        {                                                                                         \
            /* Random bytes can decode to NaN; NaN bit-encodings need not match. */               \
            if(!std::isnan(dir[i]) || !std::isnan(wrap[i]))                                       \
                EXPECT_EQ(bits_u32(wrap[i]), bits_u32(dir[i])) << "Float lane " << i;             \
            EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;                                   \
        }                                                                                         \
        (void)hipFree(dIn);                                                                       \
        (void)hipFree(dDir);                                                                      \
        (void)hipFree(dWrap);                                                                     \
        (void)hipFree(dPdir);                                                                     \
        (void)hipFree(dPwrap);                                                                    \
    }

CVT_F32_FP8_TEST(cvt_f32_fp8, 0)
CVT_F32_FP8_TEST(cvt_f32_fp8, 1)
CVT_F32_FP8_TEST(cvt_f32_fp8, 2)
CVT_F32_FP8_TEST(cvt_f32_fp8, 3)
CVT_F32_FP8_TEST(cvt_f32_bf8, 0)
CVT_F32_FP8_TEST(cvt_f32_bf8, 1)
CVT_F32_FP8_TEST(cvt_f32_bf8, 2)
CVT_F32_FP8_TEST(cvt_f32_bf8, 3)

// ---- cvt_pk_f32_fp8 / cvt_pk_f32_bf8 (packed unpack) -----------------------
template <bool WordSel>
__global__ void k_cvt_pk_f32_fp8_pair(const int*     packed,
                                      float*         direct,
                                      float*         wrapper,
                                      std::uint32_t* pay_direct,
                                      std::uint32_t* pay_wrapper)
{
    int               i = threadIdx.x;
    fpsan::v2f_native d = __builtin_amdgcn_cvt_pk_f32_fp8(packed[i], WordSel);
    auto              w = fpsan::amdgcn_cvt_pk_f32_fp8<WordSel, Semantics::Native, kCC>(packed[i]);
    direct[2 * i]       = d[0];
    direct[2 * i + 1]   = d[1];
    wrapper[2 * i]      = w.get(0).to_float();
    wrapper[2 * i + 1]  = w.get(1).to_float();

    const std::uint32_t u     = static_cast<std::uint32_t>(packed[i]);
    const std::uint8_t  byte0 = static_cast<std::uint8_t>((u >> ((2 * WordSel) * 8)) & 0xFFu);
    const std::uint8_t  byte1 = static_cast<std::uint8_t>((u >> ((2 * WordSel + 1) * 8)) & 0xFFu);
    auto fp80         = Value<fpsan::fp8_e4m3, Semantics::Triton, kCC>::from_fpsan_payload(byte0);
    auto fp81         = Value<fpsan::fp8_e4m3, Semantics::Triton, kCC>::from_fpsan_payload(byte1);
    auto wf           = fpsan::amdgcn_cvt_pk_f32_fp8<WordSel, Semantics::Triton, kCC>(packed[i]);
    pay_direct[2 * i] = fpsan::cast<float>(fp80).fpsan_payload();
    pay_direct[2 * i + 1]  = fpsan::cast<float>(fp81).fpsan_payload();
    pay_wrapper[2 * i]     = wf.get(0).fpsan_payload();
    pay_wrapper[2 * i + 1] = wf.get(1).fpsan_payload();
}

template <bool WordSel>
__global__ void k_cvt_pk_f32_bf8_pair(const int*     packed,
                                      float*         direct,
                                      float*         wrapper,
                                      std::uint32_t* pay_direct,
                                      std::uint32_t* pay_wrapper)
{
    int               i = threadIdx.x;
    fpsan::v2f_native d = __builtin_amdgcn_cvt_pk_f32_bf8(packed[i], WordSel);
    auto              w = fpsan::amdgcn_cvt_pk_f32_bf8<WordSel, Semantics::Native, kCC>(packed[i]);
    direct[2 * i]       = d[0];
    direct[2 * i + 1]   = d[1];
    wrapper[2 * i]      = w.get(0).to_float();
    wrapper[2 * i + 1]  = w.get(1).to_float();

    const std::uint32_t u     = static_cast<std::uint32_t>(packed[i]);
    const std::uint8_t  byte0 = static_cast<std::uint8_t>((u >> ((2 * WordSel) * 8)) & 0xFFu);
    const std::uint8_t  byte1 = static_cast<std::uint8_t>((u >> ((2 * WordSel + 1) * 8)) & 0xFFu);
    auto bf80         = Value<fpsan::fp8_e5m2, Semantics::Triton, kCC>::from_fpsan_payload(byte0);
    auto bf81         = Value<fpsan::fp8_e5m2, Semantics::Triton, kCC>::from_fpsan_payload(byte1);
    auto wf           = fpsan::amdgcn_cvt_pk_f32_bf8<WordSel, Semantics::Triton, kCC>(packed[i]);
    pay_direct[2 * i] = fpsan::cast<float>(bf80).fpsan_payload();
    pay_direct[2 * i + 1]  = fpsan::cast<float>(bf81).fpsan_payload();
    pay_wrapper[2 * i]     = wf.get(0).fpsan_payload();
    pay_wrapper[2 * i + 1] = wf.get(1).fpsan_payload();
}

#define CVT_PK_F32_FP8_TEST(FAMILY, WORDSEL)                                                      \
    TEST(Cvt, FAMILY##_WordSel##WORDSEL)                                                          \
    {                                                                                             \
        int ndev = 0;                                                                             \
        if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)                                   \
            GTEST_SKIP() << "no HIP device";                                                      \
        auto           in = make_packed_ints();                                                   \
        int*           dIn;                                                                       \
        float *        dDir, *dWrap;                                                              \
        std::uint32_t *dPdir, *dPwrap;                                                            \
        HIP_CHECK(hipMalloc(&dIn, kFp8N * sizeof(int)));                                          \
        HIP_CHECK(hipMalloc(&dDir, 2 * kFp8N * sizeof(float)));                                   \
        HIP_CHECK(hipMalloc(&dWrap, 2 * kFp8N * sizeof(float)));                                  \
        HIP_CHECK(hipMalloc(&dPdir, 2 * kFp8N * sizeof(std::uint32_t)));                          \
        HIP_CHECK(hipMalloc(&dPwrap, 2 * kFp8N * sizeof(std::uint32_t)));                         \
        HIP_CHECK(hipMemcpy(dIn, in.data(), kFp8N * sizeof(int), hipMemcpyHostToDevice));         \
        k_##FAMILY##_pair<WORDSEL><<<1, kFp8N>>>(dIn, dDir, dWrap, dPdir, dPwrap);                \
        HIP_CHECK(hipDeviceSynchronize());                                                        \
        std::vector<float>         dir(2 * kFp8N), wrap(2 * kFp8N);                               \
        std::vector<std::uint32_t> pdir(2 * kFp8N), pwrap(2 * kFp8N);                             \
        HIP_CHECK(hipMemcpy(dir.data(), dDir, 2 * kFp8N * sizeof(float), hipMemcpyDeviceToHost)); \
        HIP_CHECK(                                                                                \
            hipMemcpy(wrap.data(), dWrap, 2 * kFp8N * sizeof(float), hipMemcpyDeviceToHost));     \
        HIP_CHECK(hipMemcpy(                                                                      \
            pdir.data(), dPdir, 2 * kFp8N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));       \
        HIP_CHECK(hipMemcpy(                                                                      \
            pwrap.data(), dPwrap, 2 * kFp8N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));     \
        for(int i = 0; i < 2 * kFp8N; ++i)                                                        \
        {                                                                                         \
            if(!std::isnan(dir[i]) || !std::isnan(wrap[i]))                                       \
                EXPECT_EQ(bits_u32(wrap[i]), bits_u32(dir[i])) << "Float lane " << i;             \
            EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;                                   \
        }                                                                                         \
        (void)hipFree(dIn);                                                                       \
        (void)hipFree(dDir);                                                                      \
        (void)hipFree(dWrap);                                                                     \
        (void)hipFree(dPdir);                                                                     \
        (void)hipFree(dPwrap);                                                                    \
    }

CVT_PK_F32_FP8_TEST(cvt_pk_f32_fp8, false)
CVT_PK_F32_FP8_TEST(cvt_pk_f32_fp8, true)
CVT_PK_F32_FP8_TEST(cvt_pk_f32_bf8, false)
CVT_PK_F32_FP8_TEST(cvt_pk_f32_bf8, true)

// ---- cvt_pk_fp8_f32 / cvt_pk_bf8_f32 (pack) --------------------------------
template <bool DstLo>
__global__ void
    k_cvt_pk_fp8_f32_pair(const float* a, const float* b, const int* old, int* direct, int* wrapper)
{
    int i = threadIdx.x;
    // Builtin word-select is false=low, true=high; the wrapper's DstLo=true means
    // LOW, so the equivalent direct call passes !DstLo.
    direct[i] = __builtin_amdgcn_cvt_pk_fp8_f32(a[i], b[i], old[i], !DstLo);
    Value<float, Semantics::Native, kCC> av{a[i]}, bv{b[i]};
    wrapper[i] = fpsan::amdgcn_cvt_pk_fp8_f32<DstLo, Semantics::Native, kCC>(av, bv, old[i]);
}

template <bool DstLo>
__global__ void
    k_cvt_pk_bf8_f32_pair(const float* a, const float* b, const int* old, int* direct, int* wrapper)
{
    int i = threadIdx.x;
    // See cvt_pk_fp8_f32 above: wrapper DstLo=true -> low, builtin true -> high.
    direct[i] = __builtin_amdgcn_cvt_pk_bf8_f32(a[i], b[i], old[i], !DstLo);
    Value<float, Semantics::Native, kCC> av{a[i]}, bv{b[i]};
    wrapper[i] = fpsan::amdgcn_cvt_pk_bf8_f32<DstLo, Semantics::Native, kCC>(av, bv, old[i]);
}

template <bool DstLo>
__global__ void k_cvt_pk_fp8_fpsan(
    const float* a, const float* b, const int* old, int* got_expected, int* got_wrapper)
{
    int                                  i = threadIdx.x;
    Value<float, Semantics::Triton, kCC> av{a[i]}, bv{b[i]};
    auto                                 afp8 = fpsan::cast<fpsan::fp8_e4m3>(av);
    auto                                 bfp8 = fpsan::cast<fpsan::fp8_e4m3>(bv);
    std::uint8_t                         ab   = static_cast<std::uint8_t>(afp8.fpsan_payload());
    std::uint8_t                         bb   = static_cast<std::uint8_t>(bfp8.fpsan_payload());
    std::uint32_t                        u    = static_cast<std::uint32_t>(old[i]);
    if(DstLo)
        u = (u & 0xFFFF0000u) | static_cast<std::uint32_t>(ab)
            | (static_cast<std::uint32_t>(bb) << 8);
    else
        u = (u & 0x0000FFFFu) | (static_cast<std::uint32_t>(ab) << 16)
            | (static_cast<std::uint32_t>(bb) << 24);
    got_expected[i] = static_cast<int>(u);
    got_wrapper[i]  = fpsan::amdgcn_cvt_pk_fp8_f32<DstLo, Semantics::Triton, kCC>(av, bv, old[i]);
}

template <bool DstLo>
__global__ void k_cvt_pk_bf8_fpsan(
    const float* a, const float* b, const int* old, int* got_expected, int* got_wrapper)
{
    int                                  i = threadIdx.x;
    Value<float, Semantics::Triton, kCC> av{a[i]}, bv{b[i]};
    auto                                 afp8 = fpsan::cast<fpsan::fp8_e5m2>(av);
    auto                                 bfp8 = fpsan::cast<fpsan::fp8_e5m2>(bv);
    std::uint8_t                         ab   = static_cast<std::uint8_t>(afp8.fpsan_payload());
    std::uint8_t                         bb   = static_cast<std::uint8_t>(bfp8.fpsan_payload());
    std::uint32_t                        u    = static_cast<std::uint32_t>(old[i]);
    if(DstLo)
        u = (u & 0xFFFF0000u) | static_cast<std::uint32_t>(ab)
            | (static_cast<std::uint32_t>(bb) << 8);
    else
        u = (u & 0x0000FFFFu) | (static_cast<std::uint32_t>(ab) << 16)
            | (static_cast<std::uint32_t>(bb) << 24);
    got_expected[i] = static_cast<int>(u);
    got_wrapper[i]  = fpsan::amdgcn_cvt_pk_bf8_f32<DstLo, Semantics::Triton, kCC>(av, bv, old[i]);
}

#define CVT_PK_FP8_FLOAT_TEST(FAMILY, DSTLO)                                                  \
    TEST(Cvt, FAMILY##_Float_DstLo##DSTLO)                                                    \
    {                                                                                         \
        int ndev = 0;                                                                         \
        if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)                               \
            GTEST_SKIP() << "no HIP device";                                                  \
        auto   a   = make_fp8_floats();                                                       \
        auto   b   = make_fp8_floats();                                                       \
        auto   old = make_packed_ints();                                                      \
        float *dA, *dB;                                                                       \
        int *  dOld, *dDir, *dWrap;                                                           \
        HIP_CHECK(hipMalloc(&dA, kFp8N * sizeof(float)));                                     \
        HIP_CHECK(hipMalloc(&dB, kFp8N * sizeof(float)));                                     \
        HIP_CHECK(hipMalloc(&dOld, kFp8N * sizeof(int)));                                     \
        HIP_CHECK(hipMalloc(&dDir, kFp8N * sizeof(int)));                                     \
        HIP_CHECK(hipMalloc(&dWrap, kFp8N * sizeof(int)));                                    \
        HIP_CHECK(hipMemcpy(dA, a.data(), kFp8N * sizeof(float), hipMemcpyHostToDevice));     \
        HIP_CHECK(hipMemcpy(dB, b.data(), kFp8N * sizeof(float), hipMemcpyHostToDevice));     \
        HIP_CHECK(hipMemcpy(dOld, old.data(), kFp8N * sizeof(int), hipMemcpyHostToDevice));   \
        k_##FAMILY##_pair<DSTLO><<<1, kFp8N>>>(dA, dB, dOld, dDir, dWrap);                    \
        HIP_CHECK(hipDeviceSynchronize());                                                    \
        std::vector<int> dir(kFp8N), wrap(kFp8N);                                             \
        HIP_CHECK(hipMemcpy(dir.data(), dDir, kFp8N * sizeof(int), hipMemcpyDeviceToHost));   \
        HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kFp8N * sizeof(int), hipMemcpyDeviceToHost)); \
        for(int i = 0; i < kFp8N; ++i)                                                        \
            EXPECT_EQ(wrap[i], dir[i]) << "lane " << i;                                       \
        (void)hipFree(dA);                                                                    \
        (void)hipFree(dB);                                                                    \
        (void)hipFree(dOld);                                                                  \
        (void)hipFree(dDir);                                                                  \
        (void)hipFree(dWrap);                                                                 \
    }

CVT_PK_FP8_FLOAT_TEST(cvt_pk_fp8_f32, true)
CVT_PK_FP8_FLOAT_TEST(cvt_pk_fp8_f32, false)
CVT_PK_FP8_FLOAT_TEST(cvt_pk_bf8_f32, true)
CVT_PK_FP8_FLOAT_TEST(cvt_pk_bf8_f32, false)

#define CVT_PK_FP8_FPSAN_TEST(FAMILY, DSTLO)                                                  \
    TEST(Cvt, FAMILY##_Fpsan_DstLo##DSTLO)                                                    \
    {                                                                                         \
        int ndev = 0;                                                                         \
        if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)                               \
            GTEST_SKIP() << "no HIP device";                                                  \
        auto   a   = make_fp8_floats();                                                       \
        auto   b   = make_fp8_floats();                                                       \
        auto   old = make_packed_ints();                                                      \
        float *dA, *dB;                                                                       \
        int *  dOld, *dExp, *dWrap;                                                           \
        HIP_CHECK(hipMalloc(&dA, kFp8N * sizeof(float)));                                     \
        HIP_CHECK(hipMalloc(&dB, kFp8N * sizeof(float)));                                     \
        HIP_CHECK(hipMalloc(&dOld, kFp8N * sizeof(int)));                                     \
        HIP_CHECK(hipMalloc(&dExp, kFp8N * sizeof(int)));                                     \
        HIP_CHECK(hipMalloc(&dWrap, kFp8N * sizeof(int)));                                    \
        HIP_CHECK(hipMemcpy(dA, a.data(), kFp8N * sizeof(float), hipMemcpyHostToDevice));     \
        HIP_CHECK(hipMemcpy(dB, b.data(), kFp8N * sizeof(float), hipMemcpyHostToDevice));     \
        HIP_CHECK(hipMemcpy(dOld, old.data(), kFp8N * sizeof(int), hipMemcpyHostToDevice));   \
        k_##FAMILY##_fpsan<DSTLO><<<1, kFp8N>>>(dA, dB, dOld, dExp, dWrap);                   \
        HIP_CHECK(hipDeviceSynchronize());                                                    \
        std::vector<int> exp(kFp8N), wrap(kFp8N);                                             \
        HIP_CHECK(hipMemcpy(exp.data(), dExp, kFp8N * sizeof(int), hipMemcpyDeviceToHost));   \
        HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kFp8N * sizeof(int), hipMemcpyDeviceToHost)); \
        for(int i = 0; i < kFp8N; ++i)                                                        \
            EXPECT_EQ(wrap[i], exp[i]) << "lane " << i;                                       \
        (void)hipFree(dA);                                                                    \
        (void)hipFree(dB);                                                                    \
        (void)hipFree(dOld);                                                                  \
        (void)hipFree(dExp);                                                                  \
        (void)hipFree(dWrap);                                                                 \
    }

CVT_PK_FP8_FPSAN_TEST(cvt_pk_fp8, true)
CVT_PK_FP8_FPSAN_TEST(cvt_pk_fp8, false)
CVT_PK_FP8_FPSAN_TEST(cvt_pk_bf8, true)
CVT_PK_FP8_FPSAN_TEST(cvt_pk_bf8, false)

// ---- cvt_sr_fp8_f32 / cvt_sr_bf8_f32 (stochastic round + byte splice) -------
template <int ByteIdx>
__global__ void k_cvt_sr_fp8_f32_pair(const float* a,
                                      const int*   old,
                                      const int*   seed,
                                      int*         direct,
                                      int*         wrapper,
                                      int*         expected,
                                      int*         fpsan_wrapper)
{
    int i     = threadIdx.x;
    direct[i] = __builtin_amdgcn_cvt_sr_fp8_f32(a[i], old[i], seed[i], ByteIdx);
    Value<float, Semantics::Native, kCC> avf{a[i]};
    wrapper[i] = fpsan::amdgcn_cvt_sr_fp8_f32<ByteIdx, Semantics::Native, kCC>(
        avf, old[i], static_cast<std::uint32_t>(seed[i]));

    Value<float, Semantics::Triton, kCC> avp{a[i]};
    auto                                 f8 = fpsan::cast<fpsan::fp8_e4m3>(avp);
    std::uint32_t u = static_cast<std::uint32_t>(old[i]) & ~(std::uint32_t{0xFFu} << (ByteIdx * 8));
    u |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(f8.fpsan_payload())) << (ByteIdx * 8);
    expected[i]      = static_cast<int>(u);
    fpsan_wrapper[i] = fpsan::amdgcn_cvt_sr_fp8_f32<ByteIdx, Semantics::Triton, kCC>(
        avp, old[i], static_cast<std::uint32_t>(seed[i]));
}

template <int ByteIdx>
__global__ void k_cvt_sr_bf8_f32_pair(const float* a,
                                      const int*   old,
                                      const int*   seed,
                                      int*         direct,
                                      int*         wrapper,
                                      int*         expected,
                                      int*         fpsan_wrapper)
{
    int i     = threadIdx.x;
    direct[i] = __builtin_amdgcn_cvt_sr_bf8_f32(a[i], old[i], seed[i], ByteIdx);
    Value<float, Semantics::Native, kCC> avf{a[i]};
    wrapper[i] = fpsan::amdgcn_cvt_sr_bf8_f32<ByteIdx, Semantics::Native, kCC>(
        avf, old[i], static_cast<std::uint32_t>(seed[i]));

    Value<float, Semantics::Triton, kCC> avp{a[i]};
    auto                                 bf8 = fpsan::cast<fpsan::fp8_e5m2>(avp);
    std::uint32_t u = static_cast<std::uint32_t>(old[i]) & ~(std::uint32_t{0xFFu} << (ByteIdx * 8));
    u |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(bf8.fpsan_payload()))
         << (ByteIdx * 8);
    expected[i]      = static_cast<int>(u);
    fpsan_wrapper[i] = fpsan::amdgcn_cvt_sr_bf8_f32<ByteIdx, Semantics::Triton, kCC>(
        avp, old[i], static_cast<std::uint32_t>(seed[i]));
}

#define CVT_SR_FP8_TEST(FAMILY, BYTEIDX)                                                        \
    TEST(Cvt, FAMILY##_Byte##BYTEIDX)                                                           \
    {                                                                                           \
        int ndev = 0;                                                                           \
        if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)                                 \
            GTEST_SKIP() << "no HIP device";                                                    \
        auto   a    = make_fp8_floats();                                                        \
        auto   old  = make_packed_ints();                                                       \
        auto   seed = make_packed_ints();                                                       \
        float* dA;                                                                              \
        int *  dOld, *dSeed, *dDir, *dWrap, *dExp, *dFwrap;                                     \
        HIP_CHECK(hipMalloc(&dA, kFp8N * sizeof(float)));                                       \
        HIP_CHECK(hipMalloc(&dOld, kFp8N * sizeof(int)));                                       \
        HIP_CHECK(hipMalloc(&dSeed, kFp8N * sizeof(int)));                                      \
        HIP_CHECK(hipMalloc(&dDir, kFp8N * sizeof(int)));                                       \
        HIP_CHECK(hipMalloc(&dWrap, kFp8N * sizeof(int)));                                      \
        HIP_CHECK(hipMalloc(&dExp, kFp8N * sizeof(int)));                                       \
        HIP_CHECK(hipMalloc(&dFwrap, kFp8N * sizeof(int)));                                     \
        HIP_CHECK(hipMemcpy(dA, a.data(), kFp8N * sizeof(float), hipMemcpyHostToDevice));       \
        HIP_CHECK(hipMemcpy(dOld, old.data(), kFp8N * sizeof(int), hipMemcpyHostToDevice));     \
        HIP_CHECK(hipMemcpy(dSeed, seed.data(), kFp8N * sizeof(int), hipMemcpyHostToDevice));   \
        k_##FAMILY##_pair<BYTEIDX><<<1, kFp8N>>>(dA, dOld, dSeed, dDir, dWrap, dExp, dFwrap);   \
        HIP_CHECK(hipDeviceSynchronize());                                                      \
        std::vector<int> dir(kFp8N), wrap(kFp8N), exp(kFp8N), fwrap(kFp8N);                     \
        HIP_CHECK(hipMemcpy(dir.data(), dDir, kFp8N * sizeof(int), hipMemcpyDeviceToHost));     \
        HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kFp8N * sizeof(int), hipMemcpyDeviceToHost));   \
        HIP_CHECK(hipMemcpy(exp.data(), dExp, kFp8N * sizeof(int), hipMemcpyDeviceToHost));     \
        HIP_CHECK(hipMemcpy(fwrap.data(), dFwrap, kFp8N * sizeof(int), hipMemcpyDeviceToHost)); \
        for(int i = 0; i < kFp8N; ++i)                                                          \
        {                                                                                       \
            EXPECT_EQ(wrap[i], dir[i]) << "Float lane " << i;                                   \
            EXPECT_EQ(fwrap[i], exp[i]) << "FPSan lane " << i;                                  \
        }                                                                                       \
        (void)hipFree(dA);                                                                      \
        (void)hipFree(dOld);                                                                    \
        (void)hipFree(dSeed);                                                                   \
        (void)hipFree(dDir);                                                                    \
        (void)hipFree(dWrap);                                                                   \
        (void)hipFree(dExp);                                                                    \
        (void)hipFree(dFwrap);                                                                  \
    }

CVT_SR_FP8_TEST(cvt_sr_fp8_f32, 0)
CVT_SR_FP8_TEST(cvt_sr_fp8_f32, 1)
CVT_SR_FP8_TEST(cvt_sr_fp8_f32, 2)
CVT_SR_FP8_TEST(cvt_sr_fp8_f32, 3)
CVT_SR_FP8_TEST(cvt_sr_bf8_f32, 0)
CVT_SR_FP8_TEST(cvt_sr_bf8_f32, 1)
CVT_SR_FP8_TEST(cvt_sr_bf8_f32, 2)
CVT_SR_FP8_TEST(cvt_sr_bf8_f32, 3)

#endif // FPSAN_TEST_ENABLE_FP8_CVT
