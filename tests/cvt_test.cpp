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

#include <cstdint>
#include <cstring>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;

__global__ void k_cvt_pkrtz_float(const float* a, const float* b, std::uint32_t* out)
{
    int                                 i = threadIdx.x;
    Value<float, Semantics::Float, kCC> av{a[i]}, bv{b[i]};
    auto                                r = fpsan::amdgcn_cvt_pkrtz<Semantics::Float, kCC>(av, bv);
    auto                                vec = static_cast<fpsan::v2h_native>(r);
    out[i]                                  = __builtin_bit_cast(std::uint32_t, vec);
}

__global__ void k_cvt_pkrtz_fpsan(const float* a, const float* b, std::uint32_t* out)
{
    int                                 i = threadIdx.x;
    Value<float, Semantics::FPSan, kCC> av{a[i]}, bv{b[i]};
    auto                                r = fpsan::amdgcn_cvt_pkrtz<Semantics::FPSan, kCC>(av, bv);
    auto                                pay = r.fpsan_payload(); // v2u16
    out[i]                                  = __builtin_bit_cast(std::uint32_t, pay);
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
    using F = Value<float, Semantics::FPSan, kCC>;
    using H = Value<_Float16, Semantics::FPSan, kCC>;
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

// ============================================================================
// FP8 conversion intrinsic wrappers.
// ============================================================================

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
        = static_cast<float>(fpsan::amdgcn_cvt_f32_fp8<Idx, Semantics::Float, kCC>(packed[i]));
    const std::uint32_t u    = static_cast<std::uint32_t>(packed[i]);
    const std::uint8_t  byte = static_cast<std::uint8_t>((u >> (Idx * 8)) & 0xFFu);
    auto                v = Value<fpsan::fp8_e4m3, Semantics::FPSan, kCC>::from_fpsan_payload(byte);
    pay_direct[i]         = fpsan::cast<float>(v).fpsan_payload();
    pay_wrapper[i]
        = fpsan::amdgcn_cvt_f32_fp8<Idx, Semantics::FPSan, kCC>(packed[i]).fpsan_payload();
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
        = static_cast<float>(fpsan::amdgcn_cvt_f32_bf8<Idx, Semantics::Float, kCC>(packed[i]));
    const std::uint32_t u    = static_cast<std::uint32_t>(packed[i]);
    const std::uint8_t  byte = static_cast<std::uint8_t>((u >> (Idx * 8)) & 0xFFu);
    auto                v = Value<fpsan::fp8_e5m2, Semantics::FPSan, kCC>::from_fpsan_payload(byte);
    pay_direct[i]         = fpsan::cast<float>(v).fpsan_payload();
    pay_wrapper[i]
        = fpsan::amdgcn_cvt_f32_bf8<Idx, Semantics::FPSan, kCC>(packed[i]).fpsan_payload();
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

// ---- cvt_pk_fp8_f32 / cvt_pk_bf8_f32 (pack) --------------------------------
template <bool DstLo>
__global__ void
    k_cvt_pk_fp8_f32_pair(const float* a, const float* b, const int* old, int* direct, int* wrapper)
{
    int i = threadIdx.x;
    // Builtin word-select is false=low, true=high; the wrapper's DstLo=true means
    // LOW, so the equivalent direct call passes !DstLo.
    direct[i] = __builtin_amdgcn_cvt_pk_fp8_f32(a[i], b[i], old[i], !DstLo);
    Value<float, Semantics::Float, kCC> av{a[i]}, bv{b[i]};
    wrapper[i] = fpsan::amdgcn_cvt_pk_fp8_f32<DstLo, Semantics::Float, kCC>(av, bv, old[i]);
}

template <bool DstLo>
__global__ void
    k_cvt_pk_bf8_f32_pair(const float* a, const float* b, const int* old, int* direct, int* wrapper)
{
    int i = threadIdx.x;
    // See cvt_pk_fp8_f32 above: wrapper DstLo=true -> low, builtin true -> high.
    direct[i] = __builtin_amdgcn_cvt_pk_bf8_f32(a[i], b[i], old[i], !DstLo);
    Value<float, Semantics::Float, kCC> av{a[i]}, bv{b[i]};
    wrapper[i] = fpsan::amdgcn_cvt_pk_bf8_f32<DstLo, Semantics::Float, kCC>(av, bv, old[i]);
}

template <bool DstLo>
__global__ void k_cvt_pk_fp8_fpsan(
    const float* a, const float* b, const int* old, int* got_expected, int* got_wrapper)
{
    int                                 i = threadIdx.x;
    Value<float, Semantics::FPSan, kCC> av{a[i]}, bv{b[i]};
    auto                                afp8 = fpsan::cast<fpsan::fp8_e4m3>(av);
    auto                                bfp8 = fpsan::cast<fpsan::fp8_e4m3>(bv);
    std::uint8_t                        ab   = static_cast<std::uint8_t>(afp8.fpsan_payload());
    std::uint8_t                        bb   = static_cast<std::uint8_t>(bfp8.fpsan_payload());
    std::uint32_t                       u    = static_cast<std::uint32_t>(old[i]);
    if(DstLo)
        u = (u & 0xFFFF0000u) | static_cast<std::uint32_t>(ab)
            | (static_cast<std::uint32_t>(bb) << 8);
    else
        u = (u & 0x0000FFFFu) | (static_cast<std::uint32_t>(ab) << 16)
            | (static_cast<std::uint32_t>(bb) << 24);
    got_expected[i] = static_cast<int>(u);
    got_wrapper[i]  = fpsan::amdgcn_cvt_pk_fp8_f32<DstLo, Semantics::FPSan, kCC>(av, bv, old[i]);
}

template <bool DstLo>
__global__ void k_cvt_pk_bf8_fpsan(
    const float* a, const float* b, const int* old, int* got_expected, int* got_wrapper)
{
    int                                 i = threadIdx.x;
    Value<float, Semantics::FPSan, kCC> av{a[i]}, bv{b[i]};
    auto                                afp8 = fpsan::cast<fpsan::fp8_e5m2>(av);
    auto                                bfp8 = fpsan::cast<fpsan::fp8_e5m2>(bv);
    std::uint8_t                        ab   = static_cast<std::uint8_t>(afp8.fpsan_payload());
    std::uint8_t                        bb   = static_cast<std::uint8_t>(bfp8.fpsan_payload());
    std::uint32_t                       u    = static_cast<std::uint32_t>(old[i]);
    if(DstLo)
        u = (u & 0xFFFF0000u) | static_cast<std::uint32_t>(ab)
            | (static_cast<std::uint32_t>(bb) << 8);
    else
        u = (u & 0x0000FFFFu) | (static_cast<std::uint32_t>(ab) << 16)
            | (static_cast<std::uint32_t>(bb) << 24);
    got_expected[i] = static_cast<int>(u);
    got_wrapper[i]  = fpsan::amdgcn_cvt_pk_bf8_f32<DstLo, Semantics::FPSan, kCC>(av, bv, old[i]);
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
