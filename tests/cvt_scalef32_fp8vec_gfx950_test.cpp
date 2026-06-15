// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_scalef32_fp8vec_gfx950_test.cpp
//
// GPU tests for the gfx950 fp8/bf8 SCALED VECTOR cvt wrappers in
// fpsan/amdgcn_cvt.hpp -- the cousins of the scalar cvt_scalef32_f32_{fp8,bf8}
// that operate on a 16-bit word (2 fp8/bf8 bytes) at once:
//
//   unpack: cvt_scalef32_pk_{f32,f16,bf16}_{fp8,bf8}(packed, scale)  [* scale]
//   pack:   cvt_scalef32_pk_{fp8,bf8}_{f16,bf16}(old, v2, scale)     [/ scale]
//
// Two independent oracles, so any implementation that diverges from the host
// reference fails:
//
//   * UnpackAllBytes: stage fp8 byte b in word0/byte0, unpack with scale=1, and
//     require element0 == the host OCP fp8 decode (the same decode fp8_test.cpp
//     locks down) widened to the destination precision -- over all 256 bytes,
//     including the E4M3 high-exponent / NaN region the small-int tests miss.
//   * PackUnpackRoundTrip: pack a/scale, b/scale into an fp8 word then unpack
//     with the same scale -> recover (a, b). Exercises the scaled PACK
//     (divide) and the matching scaled UNPACK (multiply) together. Float mode
//     is bit-exact for values exact in fp8; FPSan mode must match the
//     payload-ring reference (which divides on pack, multiplies on unpack).
#include "fpsan/amdgcn_cvt.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC   = Conversions::Explicit;
static constexpr int         LANES = 64;

// ---------------------------------------------------------------------------
// UnpackAllBytes: every fp8/bf8 byte -> dst, scale=1, vs host OCP decode.
// ---------------------------------------------------------------------------
// One kernel per (FP8, DstFT) combo via a tag-dispatched functor.
template <class FP8, class DstFT>
struct UnpackVec;
template <>
struct UnpackVec<fpsan::fp8_e4m3, float>
{
    __device__ static float e0(int p, float s)
    {
        return fpsan::amdgcn_cvt_scalef32_pk_f32_fp8<false, Semantics::Native, kCC>(
                   p, Value<float, Semantics::Native, kCC>{s})
            .get(0)
            .to_float();
    }
};
template <>
struct UnpackVec<fpsan::fp8_e5m2, float>
{
    __device__ static float e0(int p, float s)
    {
        return fpsan::amdgcn_cvt_scalef32_pk_f32_bf8<false, Semantics::Native, kCC>(
                   p, Value<float, Semantics::Native, kCC>{s})
            .get(0)
            .to_float();
    }
};
template <>
struct UnpackVec<fpsan::fp8_e4m3, _Float16>
{
    __device__ static float e0(int p, float s)
    {
        return static_cast<float>(
            fpsan::amdgcn_cvt_scalef32_pk_f16_fp8<false, Semantics::Native, kCC>(
                p, Value<float, Semantics::Native, kCC>{s})
                .get(0)
                .to_float());
    }
};
template <>
struct UnpackVec<fpsan::fp8_e5m2, _Float16>
{
    __device__ static float e0(int p, float s)
    {
        return static_cast<float>(
            fpsan::amdgcn_cvt_scalef32_pk_f16_bf8<false, Semantics::Native, kCC>(
                p, Value<float, Semantics::Native, kCC>{s})
                .get(0)
                .to_float());
    }
};
template <>
struct UnpackVec<fpsan::fp8_e4m3, __bf16>
{
    __device__ static float e0(int p, float s)
    {
        return static_cast<float>(
            fpsan::amdgcn_cvt_scalef32_pk_bf16_fp8<false, Semantics::Native, kCC>(
                p, Value<float, Semantics::Native, kCC>{s})
                .get(0)
                .to_float());
    }
};
template <>
struct UnpackVec<fpsan::fp8_e5m2, __bf16>
{
    __device__ static float e0(int p, float s)
    {
        return static_cast<float>(
            fpsan::amdgcn_cvt_scalef32_pk_bf16_bf8<false, Semantics::Native, kCC>(
                p, Value<float, Semantics::Native, kCC>{s})
                .get(0)
                .to_float());
    }
};

template <class FP8, class DstFT>
__global__ void k_unpack_all(const int* packed, float* out)
{
    int l  = threadIdx.x; // lane l carries fp8 byte l in word0/byte0.
    out[l] = UnpackVec<FP8, DstFT>::e0(packed[l], 1.0f);
}

template <class FP8, class DstFT>
void run_unpack_all()
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<int> packed(256);
    for(int b = 0; b < 256; ++b)
        packed[b] = b; // fp8 byte b in byte0 (word0 element0)
    int*   dIn = to_dev(packed);
    float* dO;
    HIP_CHECK(hipMalloc(&dO, 256 * sizeof(float)));
    k_unpack_all<FP8, DstFT><<<1, 256>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(256);
    HIP_CHECK(hipMemcpy(got.data(), dO, 256 * sizeof(float), hipMemcpyDeviceToHost));
    for(int b = 0; b < 256; ++b)
    {
        // Host OCP decode, then widen-through the destination precision so the
        // reference matches what the hardware stores in an f16/bf16 lane.
        float dec = static_cast<float>(FP8(static_cast<std::uint8_t>(b)));
        float ref = static_cast<float>(static_cast<DstFT>(dec));
        if(std::isnan(ref))
            EXPECT_TRUE(std::isnan(got[b])) << "byte 0x" << std::hex << b;
        else
            EXPECT_EQ(got[b], ref) << "byte 0x" << std::hex << b;
    }
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

TEST(CvtScalef32Fp8Vec, UnpackAll_F32_Fp8)
{
    run_unpack_all<fpsan::fp8_e4m3, float>();
}
TEST(CvtScalef32Fp8Vec, UnpackAll_F32_Bf8)
{
    run_unpack_all<fpsan::fp8_e5m2, float>();
}
TEST(CvtScalef32Fp8Vec, UnpackAll_F16_Fp8)
{
    run_unpack_all<fpsan::fp8_e4m3, _Float16>();
}
TEST(CvtScalef32Fp8Vec, UnpackAll_F16_Bf8)
{
    run_unpack_all<fpsan::fp8_e5m2, _Float16>();
}
TEST(CvtScalef32Fp8Vec, UnpackAll_BF16_Fp8)
{
    run_unpack_all<fpsan::fp8_e4m3, __bf16>();
}
TEST(CvtScalef32Fp8Vec, UnpackAll_BF16_Bf8)
{
    run_unpack_all<fpsan::fp8_e5m2, __bf16>();
}

// ---------------------------------------------------------------------------
// PackUnpackRoundTrip: pack {a,b}/scale into an fp8 word (DstLo), then unpack
// with the same scale -> recover {a,b}. Validates the new scaled PACK (divide)
// + the matching scaled UNPACK (multiply), in both modes.
// ---------------------------------------------------------------------------
// SrcVEC is the native v2 of the source float kind; DstFT names the fp8 type.
template <class FP8, class SrcVEC>
struct PackVec;
template <>
struct PackVec<fpsan::fp8_e4m3, fpsan::v2h_native>
{
    template <Semantics S>
    __device__ static int pack(int old, Value<fpsan::v2h_native, S, kCC> v, Value<float, S, kCC> sc)
    {
        return fpsan::amdgcn_cvt_scalef32_pk_fp8_f16<true, S, kCC>(old, v, sc);
    }
    template <Semantics S>
    __device__ static Value<fpsan::v2f_native, S, kCC> unpack(int p, Value<float, S, kCC> sc)
    {
        return fpsan::amdgcn_cvt_scalef32_pk_f32_fp8<false, S, kCC>(p, sc);
    }
};
template <>
struct PackVec<fpsan::fp8_e4m3, fpsan::v2bf_native>
{
    template <Semantics S>
    __device__ static int
        pack(int old, Value<fpsan::v2bf_native, S, kCC> v, Value<float, S, kCC> sc)
    {
        return fpsan::amdgcn_cvt_scalef32_pk_fp8_bf16<true, S, kCC>(old, v, sc);
    }
    template <Semantics S>
    __device__ static Value<fpsan::v2f_native, S, kCC> unpack(int p, Value<float, S, kCC> sc)
    {
        return fpsan::amdgcn_cvt_scalef32_pk_f32_fp8<false, S, kCC>(p, sc);
    }
};
template <>
struct PackVec<fpsan::fp8_e5m2, fpsan::v2h_native>
{
    template <Semantics S>
    __device__ static int pack(int old, Value<fpsan::v2h_native, S, kCC> v, Value<float, S, kCC> sc)
    {
        return fpsan::amdgcn_cvt_scalef32_pk_bf8_f16<true, S, kCC>(old, v, sc);
    }
    template <Semantics S>
    __device__ static Value<fpsan::v2f_native, S, kCC> unpack(int p, Value<float, S, kCC> sc)
    {
        return fpsan::amdgcn_cvt_scalef32_pk_f32_bf8<false, S, kCC>(p, sc);
    }
};
template <>
struct PackVec<fpsan::fp8_e5m2, fpsan::v2bf_native>
{
    template <Semantics S>
    __device__ static int
        pack(int old, Value<fpsan::v2bf_native, S, kCC> v, Value<float, S, kCC> sc)
    {
        return fpsan::amdgcn_cvt_scalef32_pk_bf8_bf16<true, S, kCC>(old, v, sc);
    }
    template <Semantics S>
    __device__ static Value<fpsan::v2f_native, S, kCC> unpack(int p, Value<float, S, kCC> sc)
    {
        return fpsan::amdgcn_cvt_scalef32_pk_f32_bf8<false, S, kCC>(p, sc);
    }
};

template <Semantics S, class FP8, class SrcVEC, class SrcElem, class Out>
__global__ void k_roundtrip(const float* in, Out* out, float scale)
{
    int                   l = threadIdx.x;
    Value<SrcVEC, S, kCC> v{};
    v.set(0, Value<SrcElem, S, kCC>{static_cast<SrcElem>(in[2 * l])});
    v.set(1, Value<SrcElem, S, kCC>{static_cast<SrcElem>(in[2 * l + 1])});
    Value<float, S, kCC> sc{scale};
    int                  packed = PackVec<FP8, SrcVEC>::template pack<S>(0, v, sc);
    auto                 r      = PackVec<FP8, SrcVEC>::template unpack<S>(packed, sc);
    if constexpr(S == Semantics::Native)
    {
        out[2 * l]     = r.get(0).to_float();
        out[2 * l + 1] = r.get(1).to_float();
    }
    else
    {
        out[2 * l]     = r.get(0).fpsan_payload();
        out[2 * l + 1] = r.get(1).fpsan_payload();
    }
}

template <class FP8, class SrcVEC, class SrcElem>
void run_roundtrip()
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    const float        scale = 2.0f;
    std::vector<float> in(2 * LANES);
    std::mt19937       rng = fpsan_test::make_rng();
    for(auto& x : in)
        x = fpsan_test::pick_int_valued<float>(rng, -4, 4); // x/2 exact in fp8
    float* dIn = to_dev(in);
    // Float: round-trip recovers the input exactly for exact-in-fp8 values.
    {
        float* dO;
        HIP_CHECK(hipMalloc(&dO, 2 * LANES * sizeof(float)));
        k_roundtrip<Semantics::Native, FP8, SrcVEC, SrcElem, float><<<1, LANES>>>(dIn, dO, scale);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> got(2 * LANES);
        HIP_CHECK(hipMemcpy(got.data(), dO, 2 * LANES * sizeof(float), hipMemcpyDeviceToHost));
        for(int i = 0; i < 2 * LANES; ++i)
            EXPECT_EQ(got[i], in[i]) << "Float round-trip at " << i;
        (void)hipFree(dO);
    }
    // FPSan: payload must equal the payload-ring reference. Pack divides, unpack
    // multiplies: cast<float>(cast<FP8>(cast<SrcElem>(x) / scale)) * scale.
    {
        using VF = Value<float, Semantics::Triton, kCC>;
        using VS = Value<SrcElem, Semantics::Triton, kCC>;
        std::vector<std::uint32_t> ref(2 * LANES);
        for(int i = 0; i < 2 * LANES; ++i)
        {
            VS   src{static_cast<SrcElem>(in[i])};
            auto packed = fpsan::cast<FP8>(fpsan::cast<float>(src) / VF(scale));
            ref[i]      = (fpsan::cast<float>(packed) * VF(scale)).fpsan_payload();
        }
        std::uint32_t* dO;
        HIP_CHECK(hipMalloc(&dO, 2 * LANES * sizeof(std::uint32_t)));
        k_roundtrip<Semantics::Triton, FP8, SrcVEC, SrcElem, std::uint32_t>
            <<<1, LANES>>>(dIn, dO, scale);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<std::uint32_t> got(2 * LANES);
        HIP_CHECK(
            hipMemcpy(got.data(), dO, 2 * LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
        for(int i = 0; i < 2 * LANES; ++i)
            EXPECT_EQ(got[i], ref[i]) << "FPSan payload at " << i;
        (void)hipFree(dO);
    }
    (void)hipFree(dIn);
}

TEST(CvtScalef32Fp8Vec, RoundTrip_Fp8_F16)
{
    run_roundtrip<fpsan::fp8_e4m3, fpsan::v2h_native, _Float16>();
}
TEST(CvtScalef32Fp8Vec, RoundTrip_Fp8_BF16)
{
    run_roundtrip<fpsan::fp8_e4m3, fpsan::v2bf_native, __bf16>();
}
TEST(CvtScalef32Fp8Vec, RoundTrip_Bf8_F16)
{
    run_roundtrip<fpsan::fp8_e5m2, fpsan::v2h_native, _Float16>();
}
TEST(CvtScalef32Fp8Vec, RoundTrip_Bf8_BF16)
{
    run_roundtrip<fpsan::fp8_e5m2, fpsan::v2bf_native, __bf16>();
}
