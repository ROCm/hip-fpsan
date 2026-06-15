// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/wmma_16x16x128_f8f6f4_gfx1250_test.cpp
//
// gfx1250 wmma_f32_16x16x128_f8f6f4: a single K=128 WMMA whose A and B operands
// each carry a per-operand FORMAT immediate selecting how the fixed 512-bit
// (v16i32) per-lane fragment is interpreted: FP8(0)/BF8(1) 8-bit, FP6(2)/BF6(3)
// 6-bit, FP4(4) 4-bit. Unlike the dedicated fp8_fp8/.../bf8_bf8 builtins (8-bit
// only), this one reaches the sub-byte formats and mixed A/B formats.
//
// This file's GROUNDING test pins the still-unverified sub-byte fragment layout.
// Hypothesis (grounded in the AMD math-engine model and continuous with the
// validated 8-bit K=128 layout): the cross-lane element->(lane, slot) mapping is
// the SAME Wmma16x16x128Layout used by the fp8 K=128 WMMA -- only the in-lane
// packing changes, slot s occupying the contiguous Width-bit field at bit
// Width*s of the 512-bit register. The test stages A/B that way, runs the real
// builtin (Float oracle), and asserts bit-exactness vs an independent host
// matmul on exact-integer inputs. Equality across distinct random inputs and all
// five formats (incl. mixed A/B) can only hold if the staged layout equals the
// hardware fragment ABI.
//
// Requires gfx1250 hardware; built only under the gfx1250 test tier.
#include "fpsan/amdgcn_matrix.hpp"
#include "fpsan/detail/fp8.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;
using fpsan::detail::f32_to_narrow;
using fpsan::detail::FpFormat;
using fpsan::detail::kBf6E3M2;
using fpsan::detail::kFp4E2M1;
using fpsan::detail::kFp6E2M3;
using fpsan::detail::kFp8E4M3;
using fpsan::detail::kFp8E5M2;

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4)

static constexpr Conversions kCC = Conversions::Explicit;
static constexpr int         M = 16, N = 16, K = 128;

using v16i32_native = int __attribute__((ext_vector_type(16)));
using v8f_native    = float __attribute__((ext_vector_type(8)));

// f8f6f4 format immediate -> (host FpFormat, bit width).
static const FpFormat& fmt_of(int code)
{
    switch(code)
    {
    case 0:
        return kFp8E4M3;
    case 1:
        return kFp8E5M2;
    case 2:
        return kFp6E2M3;
    case 3:
        return kBf6E3M2;
    default:
        return kFp4E2M1;
    }
}
static int width_of(int code)
{
    return code <= 1 ? 8 : code <= 3 ? 6 : 4;
}

// Wmma16x16x128 element->fragment mapping (validated for 8-bit; hypothesised to
// hold for sub-byte with bit position = width*slot).
static int ab_lane(int row_or_col, int k)
{
    return row_or_col + 16 * ((k >> 2) & 1);
}
static int ab_slot(int k)
{
    const int reg = ((k >> 1) & 1) + 2 * ((k >> 3) & 1) + 4 * ((k >> 4) & 1) + 8 * ((k >> 5) & 1)
                    + 16 * ((k >> 6) & 1);
    return 2 * reg + (k & 1);
}

// Place a Width-bit code at slot s (bit width*s) of a 512-bit (16xu32) register.
static void pack_field(std::uint32_t* reg /*[16]*/, int slot, std::uint32_t code, int width)
{
    const int p = width * slot;
    for(int b = 0; b < width; ++b)
        if((code >> b) & 1u)
            reg[(p + b) >> 5] |= 1u << ((p + b) & 31);
}

// Build the 32-lane packed A and B operands from logical float matrices, using
// the hypothesised layout and the given per-operand format.
static void stage_operands(const std::vector<float>&   A, // [M*K], row-major A[m][k]
                           const std::vector<float>&   B, // [K*N], row-major B[k][n]
                           int                         afmt,
                           int                         bfmt,
                           std::vector<std::uint32_t>& Apack, // [32*16]
                           std::vector<std::uint32_t>& Bpack)
{
    Apack.assign(32 * 16, 0u);
    Bpack.assign(32 * 16, 0u);
    const FpFormat& af = fmt_of(afmt);
    const FpFormat& bf = fmt_of(bfmt);
    const int       aw = width_of(afmt), bw = width_of(bfmt);
    for(int m = 0; m < M; ++m)
        for(int k = 0; k < K; ++k)
        {
            const int           lane = ab_lane(m, k), slot = ab_slot(k);
            const std::uint32_t code = f32_to_narrow(A[m * K + k], af);
            pack_field(&Apack[lane * 16], slot, code, aw);
        }
    for(int n = 0; n < N; ++n)
        for(int k = 0; k < K; ++k)
        {
            const int           lane = ab_lane(n, k), slot = ab_slot(k);
            const std::uint32_t code = f32_to_narrow(B[k * N + n], bf);
            pack_field(&Bpack[lane * 16], slot, code, bw);
        }
}

// Mixed 8-bit x sub-byte physical k order (the sub operand only): reverse-
// engineered from one-hot probes, verified 128/128 both operand roles. The 8-bit
// operand keeps ab_lane/ab_slot; the sub operand uses these.
static int mix_lane(int rc, int k)
{
    return rc + 16 * ((k >> 5) & 1);
}
static int mix_slot(int k)
{
    return 32 * ((k >> 6) & 1) + 16 * ((k >> 2) & 1) + 8 * ((k >> 4) & 1) + 4 * ((k >> 3) & 1)
           + 2 * ((k >> 1) & 1) + (k & 1);
}

// Stage operands for the MIXED case (exactly one 8-bit, one sub-byte): the 8-bit
// operand on the validated ab layout, the sub-byte operand on the mixed order.
template <int AFMT, int BFMT>
static void stage_operands_mixed(const std::vector<float>&   A,
                                 const std::vector<float>&   B,
                                 std::vector<std::uint32_t>& Apack,
                                 std::vector<std::uint32_t>& Bpack)
{
    Apack.assign(32 * 16, 0u);
    Bpack.assign(32 * 16, 0u);
    const bool aSub = AFMT >= 2, bSub = BFMT >= 2;
    const int  aw = width_of(AFMT), bw = width_of(BFMT);
    for(int m = 0; m < M; ++m)
        for(int k = 0; k < K; ++k)
        {
            const int laneA = aSub ? mix_lane(m, k) : ab_lane(m, k);
            const int slotA = aSub ? mix_slot(k) : ab_slot(k);
            pack_field(&Apack[laneA * 16], slotA, f32_to_narrow(A[m * K + k], fmt_of(AFMT)), aw);
        }
    for(int n = 0; n < N; ++n)
        for(int k = 0; k < K; ++k)
        {
            const int laneB = bSub ? mix_lane(n, k) : ab_lane(n, k);
            const int slotB = bSub ? mix_slot(k) : ab_slot(k);
            pack_field(&Bpack[laneB * 16], slotB, f32_to_narrow(B[k * N + n], fmt_of(BFMT)), bw);
        }
}

// One kernel per (afmt,bfmt) -- the format immediates must be compile-time. A
// template with non-type params instantiates the right builtin call.
template <int AFMT, int BFMT>
__global__ void
    k_f8f6f4(const std::uint32_t* Apack, const std::uint32_t* Bpack, const float* C, float* D)
{
    const int     lane = threadIdx.x;
    v16i32_native a, b;
    for(int i = 0; i < 16; ++i)
    {
        a[i] = static_cast<int>(Apack[lane * 16 + i]);
        b[i] = static_cast<int>(Bpack[lane * 16 + i]);
    }
    v8f_native c;
    for(int e = 0; e < 8; ++e)
        c[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
    v8f_native d = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(AFMT, a, BFMT, b, (short)0, c);
    for(int e = 0; e < 8; ++e)
        D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d[e];
}

// fp4 grid: representable exactly in every f8f6f4 format AND in f32, so the
// matmul is exact and bit-equality is well-defined.
static const float kGrid[]
    = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

template <int AFMT, int BFMT, bool MIXED = false>
static void run_layout(const char* tag)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::mt19937                       rng = fpsan_test::make_rng();
    std::uniform_int_distribution<int> pick(0, (int)(sizeof(kGrid) / sizeof(float)) - 1);
    std::vector<float>                 A(M * K), B(K * N), C(M * N);
    for(auto& x : A)
        x = kGrid[pick(rng)];
    for(auto& x : B)
        x = kGrid[pick(rng)];
    for(auto& x : C)
        x = kGrid[pick(rng)];

    std::vector<std::uint32_t> Apack, Bpack;
    if constexpr(MIXED)
        stage_operands_mixed<AFMT, BFMT>(A, B, Apack, Bpack);
    else
        stage_operands(A, B, AFMT, BFMT, Apack, Bpack);

    std::vector<float> ref(M * N);
    for(int m = 0; m < M; ++m)
        for(int n = 0; n < N; ++n)
        {
            float acc = C[m * N + n];
            for(int k = 0; k < K; ++k)
                acc += A[m * K + k] * B[k * N + n];
            ref[m * N + n] = acc;
        }

    std::uint32_t* dA = to_dev(Apack);
    std::uint32_t* dB = to_dev(Bpack);
    float*         dC = to_dev(C);
    float*         dD = nullptr;
    HIP_CHECK(hipMalloc(&dD, M * N * sizeof(float)));
    k_f8f6f4<AFMT, BFMT><<<1, 32>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(M * N);
    HIP_CHECK(hipMemcpy(got.data(), dD, M * N * sizeof(float), hipMemcpyDeviceToHost));
    for(int i = 0; i < M * N; ++i)
        EXPECT_EQ(bits_of(got[i]), bits_of(ref[i]))
            << tag << " mismatch at " << (i / N) << "," << (i % N) << " got=" << got[i]
            << " ref=" << ref[i];
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

TEST(WmmaF8f6f4_128, Fp8Fp8)
{
    run_layout<0, 0>("fp8_fp8");
}
TEST(WmmaF8f6f4_128, Bf8Bf8)
{
    run_layout<1, 1>("bf8_bf8");
}
TEST(WmmaF8f6f4_128, Fp6Fp6)
{
    run_layout<2, 2>("fp6_fp6");
}
TEST(WmmaF8f6f4_128, Bf6Bf6)
{
    run_layout<3, 3>("bf6_bf6");
}
TEST(WmmaF8f6f4_128, Fp4Fp4)
{
    run_layout<4, 4>("fp4_fp4");
}
// Mixed sub-byte widths (fp6 x fp4): all sub-byte formats share ONE physical
// k->slot order, so the unified layout pairs them correctly. Validated.
TEST(WmmaF8f6f4_128, Fp6Fp4)
{
    run_layout<2, 4>("fp6_fp4");
}

// Mixed 8-bit x sub-byte (fp8 x fp6, fp4 x fp8): the 8-bit operand and the
// sub-byte operand use DIFFERENT physical k->slot orderings on gfx1250. The
// sub-byte order was reverse-engineered from one-hot probes (mix_lane/mix_slot,
// verified 128/128 in both operand roles); staging the 8-bit operand on the
// validated ab layout and the sub operand on the mixed order makes the builtin
// match an independent host matmul bit-for-bit. Grounds the mixed mix model.
TEST(WmmaF8f6f4_128, Fp8Fp6_mixed)
{
    run_layout<0, 2, true>("fp8_fp6");
}
TEST(WmmaF8f6f4_128, Fp4Fp8_mixed)
{
    run_layout<4, 0, true>("fp4_fp8");
}
TEST(WmmaF8f6f4_128, Bf8Fp4_mixed)
{
    run_layout<1, 4, true>("bf8_fp4");
}
TEST(WmmaF8f6f4_128, Fp6Fp8_mixed)
{
    run_layout<2, 0, true>("fp6_fp8");
}

// ===================== FPSan payload property =====================
// The shipped sub-byte wrapper's FPSan dataflow must equal an independent
// payload-ring reference. The per-slot payload of a sub-byte code is the signed
// resize of its Width-bit field (the ExtSI / subbyte_widen model, same as the
// fp4/fp6 cvt-unpack path); products and the C term accumulate in the ring.
using VF = Value<float, Semantics::Triton, kCC>;

static std::uint32_t sign_resize(std::uint32_t code, int width)
{
    const std::int32_t e = static_cast<std::int32_t>(code << (32 - width)) >> (32 - width);
    return static_cast<std::uint32_t>(e);
}

template <int AFMT, int BFMT, bool MIXED = false>
__global__ void k_f8f6f4_fpsan(const std::uint32_t* Apack,
                               const std::uint32_t* Bpack,
                               const float*         C,
                               std::uint32_t*       Dpay)
{
    const int     lane = threadIdx.x;
    v16i32_native a, b;
    for(int i = 0; i < 16; ++i)
    {
        a[i] = static_cast<int>(Apack[lane * 16 + i]);
        b[i] = static_cast<int>(Bpack[lane * 16 + i]);
    }
    v8f_native cn;
    for(int e = 0; e < 8; ++e)
        cn[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
    Value<v8f_native, Semantics::Triton, kCC> c(cn);
    Value<v8f_native, Semantics::Triton, kCC> d;
    if constexpr(MIXED)
        d = fpsan::amdgcn_wmma_f32_16x16x128_f8f6f4_mixed<AFMT, BFMT, Semantics::Triton, kCC>(
            a, b, c);
    else
        d = fpsan::amdgcn_wmma_f32_16x16x128_f8f6f4_sub<AFMT, BFMT, Semantics::Triton, kCC>(
            a, b, c);
    for(int e = 0; e < 8; ++e)
        Dpay[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).fpsan_payload();
}

template <int AFMT, int BFMT, bool MIXED = false>
static void run_fpsan(const char* tag)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::mt19937                       rng = fpsan_test::make_rng();
    std::uniform_int_distribution<int> pick(0, (int)(sizeof(kGrid) / sizeof(float)) - 1);
    std::vector<float>                 A(M * K), B(K * N), C(M * N);
    for(auto& x : A)
        x = kGrid[pick(rng)];
    for(auto& x : B)
        x = kGrid[pick(rng)];
    for(auto& x : C)
        x = kGrid[pick(rng)];

    std::vector<std::uint32_t> Apack, Bpack;
    if constexpr(MIXED)
        stage_operands_mixed<AFMT, BFMT>(A, B, Apack, Bpack);
    else
        stage_operands(A, B, AFMT, BFMT, Apack, Bpack);

    const int                  aw = width_of(AFMT), bw = width_of(BFMT);
    const std::uint32_t        amask = (1u << aw) - 1u, bmask = (1u << bw) - 1u;
    std::vector<std::uint32_t> ref(M * N);
    for(int m = 0; m < M; ++m)
        for(int n = 0; n < N; ++n)
        {
            VF acc(C[m * N + n]);
            for(int k = 0; k < K; ++k)
            {
                const std::uint32_t ca = f32_to_narrow(A[m * K + k], fmt_of(AFMT)) & amask;
                const std::uint32_t cb = f32_to_narrow(B[k * N + n], fmt_of(BFMT)) & bmask;
                acc                    = acc
                      + VF::from_fpsan_payload(sign_resize(ca, aw))
                            * VF::from_fpsan_payload(sign_resize(cb, bw));
            }
            ref[m * N + n] = acc.fpsan_payload();
        }

    std::uint32_t* dA = to_dev(Apack);
    std::uint32_t* dB = to_dev(Bpack);
    float*         dC = to_dev(C);
    std::uint32_t* dD = nullptr;
    HIP_CHECK(hipMalloc(&dD, M * N * sizeof(std::uint32_t)));
    k_f8f6f4_fpsan<AFMT, BFMT, MIXED><<<1, 32>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(M * N);
    HIP_CHECK(hipMemcpy(got.data(), dD, M * N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < M * N; ++i)
        EXPECT_EQ(got[i], ref[i]) << tag << " payload mismatch at " << (i / N) << "," << (i % N);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

TEST(WmmaF8f6f4_128, Fpsan_Fp6Fp6)
{
    run_fpsan<2, 2>("fp6_fp6");
}
TEST(WmmaF8f6f4_128, Fpsan_Bf6Bf6)
{
    run_fpsan<3, 3>("bf6_bf6");
}
TEST(WmmaF8f6f4_128, Fpsan_Fp4Fp4)
{
    run_fpsan<4, 4>("fp4_fp4");
}
TEST(WmmaF8f6f4_128, Fpsan_Fp6Fp4)
{
    run_fpsan<2, 4>("fp6_fp4");
}
// Mixed 8-bit x sub-byte FPSan: payload dataflow uses the mixed mix model
// (8-bit operand on ab layout, sub operand on mix_lane/mix_slot).
TEST(WmmaF8f6f4_128, Fpsan_Fp8Fp6_mixed)
{
    run_fpsan<0, 2, true>("fp8_fp6");
}
TEST(WmmaF8f6f4_128, Fpsan_Fp4Fp8_mixed)
{
    run_fpsan<4, 0, true>("fp4_fp8");
}
TEST(WmmaF8f6f4_128, Fpsan_Bf8Fp4_mixed)
{
    run_fpsan<1, 4, true>("bf8_fp4");
}

#endif // __has_builtin(__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4)
