// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// examples/reduction_demo.cpp
//
// Demo: sum N=256 values three different ways on the GPU, showing that
// Float-mode results disagree (because f32 add is not associative) while FPSan
// results all agree bit-for-bit (because the FPSan payload ring is integer add
// mod 2^32 -- associative + commutative -- so the reduction tree shape does not
// matter).
//
//   (1) CPU naive:   sequential f32 accumulator on the host, left-to-right.
//   (2) wfred:       each lane sums N/warpSize inputs, then
//                    wave_reduce_fadd_f32 collapses the per-lane partials via a
//                    hardware butterfly.
//   (3) matrix:      reshape inputs as a 16x16 matrix B, multiply by A=ones,
//                    yielding 16 column sums in the f32 accumulator; host then
//                    sums those 16 column sums sequentially.
//
// The matrix path is SINGLE-SOURCE but uses different hardware instructions per
// GPU architecture, selected by a device-arch `#if` (not a build flag):
//   * RDNA4 / gfx12  -> WMMA  (amdgcn_wmma_f32_16x16x16_bf16_w32, wave32)
//   * CDNA4 / gfx950 -> MFMA  (amdgcn_mfma_f32_16x16x16bf16_1k,   wave64)
// An architecture with no matrix code path here is a hard compile error. The
// host launches every kernel at the device's runtime `warpSize`, so the same
// binary drives wave32 and wave64 without a build-time wave size.
//
// All three paths cast every input to bf16 before accumulating in f32 (the
// matrix path is bf16-in/f32-out, so this is what it has to do; matching that
// cast in the other two paths means all three are summing the SAME set of
// payloads, just in different orders -- exactly what we want to exhibit).
//
// Input pattern is intentionally minimal: a single large value at index 0
// (2^24) and 1.0 at every other index.  ULP at 2^24 in f32 is 2, so a +1
// added to an accumulator already at 2^24 rounds to 2^24 (lost); a +1 added
// to a small accumulator survives.  The three paths put the +1s in
// different positions relative to the big value, so they round differently.
//
// Why bf16 (not f32, not f16): RDNA4 has no F32_F32 WMMA at 16x16x16; the
// matrix path has to narrow A/B to f16/bf16/fp8.  bf16 keeps f32's 8-bit
// exponent so the chosen pattern (2^24, 1) is bf16-exact and the cast is a
// no-op for these values -- the cast machinery is exercised, but the input
// precision isn't a moving part.
//
// Returns nonzero from main() if the predictions don't hold (Float results
// all agree, or FPSan results disagree), so the example is also a
// load-bearing ctest that catches regressions in the demonstrated invariants.

#include "fpsan/amdgcn_matrix.hpp"
#include "fpsan/amdgcn_mfma.hpp"
#include "fpsan/amdgcn_wave.hpp"
#include "fpsan/fpsan.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Pick the matrix code path from the device architecture. The arch macros are
// only defined in the device compilation pass; the host pass (no __gfx*__) falls
// through to the WMMA branch purely so the source type-checks -- the kernel only
// ever runs on the arch it was compiled for. A device arch with no matrix path
// is a hard error rather than a silent omission.
// ---------------------------------------------------------------------------
#if defined(__gfx950__)
#define DEMO_PATH_MFMA 1
#elif defined(__gfx1200__) || defined(__gfx1201__) || !defined(__HIP_DEVICE_COMPILE__)
#define DEMO_PATH_WMMA 1
#else
#error \
    "reduction_demo: no matrix code path for this GPU architecture (have: gfx12 WMMA, gfx950 MFMA)"
#endif

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr int         N   = 256;
static constexpr int         NN  = 16;
static constexpr Conversions kCC = Conversions::Explicit;

#define HIP_CHECK(e)                                                        \
    do                                                                      \
    {                                                                       \
        hipError_t e_ = (e);                                                \
        if(e_ != hipSuccess)                                                \
        {                                                                   \
            std::fprintf(stderr, "HIP error: %s\n", hipGetErrorString(e_)); \
            std::exit(1);                                                   \
        }                                                                   \
    } while(0)

// ---------------------------------------------------------------------------
// (1) CPU naive: sequential ring/float accumulator, left-to-right.
// ---------------------------------------------------------------------------
template <Semantics S>
auto cpu_naive(const float* x)
{
    using F = Value<float, S, kCC>;
    using B = Value<__bf16, S, kCC>;
    F acc{0.f};
    for(int i = 0; i < N; ++i)
        acc = acc + fpsan::cast<float>(B(static_cast<__bf16>(x[i])));
    if constexpr(S == Semantics::Float)
        return static_cast<float>(acc);
    else
        return acc.fpsan_payload();
}

// ---------------------------------------------------------------------------
// (2) wfred: per-lane partial sum of N/warpSize inputs, then wave_reduce_fadd.
//     Wave-size agnostic: the per-lane chunk and the butterfly both follow the
//     launch's wave width.
// ---------------------------------------------------------------------------
template <Semantics S, class Out>
__global__ void k_wfred(const float* x, Out* out)
{
    using F        = Value<float, S, kCC>;
    using B        = Value<__bf16, S, kCC>;
    const int lane = threadIdx.x;
    const int per  = N / static_cast<int>(blockDim.x);
    F         acc{0.f};
    for(int i = 0; i < per; ++i)
        acc = acc + fpsan::cast<float>(B(static_cast<__bf16>(x[lane * per + i])));
    auto sum = fpsan::amdgcn_wave_reduce_fadd_f32<0>(acc);
    if(lane == 0)
    {
        if constexpr(S == Semantics::Float)
            *out = static_cast<float>(sum);
        else
            *out = sum.fpsan_payload();
    }
}

// ---------------------------------------------------------------------------
// (3) matrix: column sums of B (16x16 inputs as bf16) via D = A*B with A=ones.
//     With A all-ones, D[m][n] = sum_k B[k][n] = column-n sum (same for all m).
//     For both WMMA (gfx12) and MFMA (gfx950) the 16x16x16 output places D[0][n]
//     on lane n, register 0 -- so the read-back is identical across archs; only
//     the fragment layout/fill and the instruction differ.
// ---------------------------------------------------------------------------
#if defined(DEMO_PATH_WMMA)
// Inverse of Wmma16x16x16Layout's k(lane, idx) for the A/B fragments: given the
// lane and fragment slot idx, which logical K-index it holds.
__device__ inline int wmma_frag_k(int lane, int idx)
{
    int reg = idx >> 1, half = idx & 1;
    return half | ((reg & 1) << 1) | ((lane >> 4) << 2) | ((reg >> 1) << 3);
}
#endif

template <Semantics S, class Out>
__global__ void k_matrix_column_sums(const float* x, Out* col_out)
{
    const int lane = threadIdx.x;
#if defined(DEMO_PATH_MFMA)
    // CDNA4 / gfx950: MFMA 16x16x16 bf16 (wave64, 4 bf16/lane). Fill A=ones and
    // B=inputs by walking the logical (outer, k) indices through the MFMA layout
    // helper input_loc; A[o][k] and B[k][o] share a fragment slot (M==N).
    using V4B = Value<fpsan::v4bf_native, S, kCC>;
    using V4F = Value<fpsan::v4f_native, S, kCC>;
    fpsan::v4bf_native an{}, bn{};
    for(int o = 0; o < NN; ++o)
        for(int k = 0; k < NN; ++k)
        {
            auto loc = fpsan::detail::input_loc(NN, NN, 1, o, k, 0, /*data_bits=*/16);
            if(loc.lane == lane)
            {
                const int e = 2 * loc.reg + loc.sub;
                an[e]       = static_cast<__bf16>(1.0f);
                bn[e]       = static_cast<__bf16>(x[k * NN + o]);
            }
        }
    V4B  a{an}, b{bn};
    V4F  c{fpsan::v4f_native{}};
    auto d = fpsan::amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c);
#else
    // RDNA4 / gfx12: WMMA 16x16x16 bf16 (wave32, 8 bf16/lane).
    using V8B = Value<fpsan::v8bf_native, S, kCC>;
    using V8F = Value<fpsan::v8f_native, S, kCC>;
    fpsan::v8bf_native an{}, bn{};
    for(int idx = 0; idx < 8; ++idx)
    {
        int k   = wmma_frag_k(lane, idx);
        an[idx] = static_cast<__bf16>(1.0f);
        bn[idx] = static_cast<__bf16>(x[k * NN + (lane & 15)]);
    }
    V8B  a{an}, b{bn};
    V8F  c{fpsan::v8f_native{}};
    auto d = fpsan::amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#endif
    // Lane n in 0..15 holds D[0][n] = column-n sum in accumulator register 0.
    if(lane < NN)
    {
        if constexpr(S == Semantics::Float)
            col_out[lane] = static_cast<float>(d.get(0));
        else
            col_out[lane] = d.get(0).fpsan_payload();
    }
}

// Final sequential sum of 16 column sums on the host (same shape in both
// modes -- the matrix path's distinctive accumulator order lives in the matrix
// instruction itself, not in this host sum-of-columns step).
template <Semantics S, class In>
auto host_sum16(const In* cols)
{
    using F = Value<float, S, kCC>;
    F acc{0.f};
    for(int i = 0; i < NN; ++i)
    {
        F v{0.f};
        if constexpr(S == Semantics::Float)
            v = F{cols[i]};
        else
            v = F::from_fpsan_payload(cols[i]);
        acc = acc + v;
    }
    if constexpr(S == Semantics::Float)
        return static_cast<float>(acc);
    else
        return acc.fpsan_payload();
}

static constexpr int kLabelW = 32;
static void          show_float(const char* label, float f)
{
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof bits);
    std::printf("  %-*s %14.1f  (0x%08x)\n", kLabelW, label, f, bits);
}
static void show_payload(const char* label, std::uint32_t p)
{
    std::printf("  %-*s             0x%08x\n", kLabelW, label, p);
}

int main()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    {
        std::fprintf(stderr, "no HIP device; skipping demo\n");
        return 0; // skip
    }

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    const int wave = props.warpSize; // 32 on RDNA (WMMA), 64 on CDNA (MFMA)

    // --- Inputs: one large value (2^24) at index 0; 1.0 everywhere else.
    // Both values are bf16-exact, so the bf16 cast is a no-op for the values
    // themselves and the cast machinery is exercised without adding noise.
    std::vector<float> input(N);
    input[0] = static_cast<float>(1u << 24);
    for(int i = 1; i < N; ++i)
        input[i] = 1.0f;

    float* d_in;
    HIP_CHECK(hipMalloc(&d_in, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, input.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // --- Float mode.
    const float cpu_f = cpu_naive<Semantics::Float>(input.data());

    float* d_wfred_f;
    HIP_CHECK(hipMalloc(&d_wfred_f, sizeof(float)));
    k_wfred<Semantics::Float><<<1, wave>>>(d_in, d_wfred_f);
    HIP_CHECK(hipDeviceSynchronize());
    float wfred_f = 0;
    HIP_CHECK(hipMemcpy(&wfred_f, d_wfred_f, sizeof(float), hipMemcpyDeviceToHost));

    float* d_cols_f;
    HIP_CHECK(hipMalloc(&d_cols_f, NN * sizeof(float)));
    k_matrix_column_sums<Semantics::Float><<<1, wave>>>(d_in, d_cols_f);
    HIP_CHECK(hipDeviceSynchronize());
    float cols_f[NN]{};
    HIP_CHECK(hipMemcpy(cols_f, d_cols_f, NN * sizeof(float), hipMemcpyDeviceToHost));
    const float wmma_f = host_sum16<Semantics::Float>(cols_f);

    // --- FPSan mode.
    const std::uint32_t cpu_p = cpu_naive<Semantics::FPSan>(input.data());

    std::uint32_t* d_wfred_p;
    HIP_CHECK(hipMalloc(&d_wfred_p, sizeof(std::uint32_t)));
    k_wfred<Semantics::FPSan><<<1, wave>>>(d_in, d_wfred_p);
    HIP_CHECK(hipDeviceSynchronize());
    std::uint32_t wfred_p = 0;
    HIP_CHECK(hipMemcpy(&wfred_p, d_wfred_p, sizeof(std::uint32_t), hipMemcpyDeviceToHost));

    std::uint32_t* d_cols_p;
    HIP_CHECK(hipMalloc(&d_cols_p, NN * sizeof(std::uint32_t)));
    k_matrix_column_sums<Semantics::FPSan><<<1, wave>>>(d_in, d_cols_p);
    HIP_CHECK(hipDeviceSynchronize());
    std::uint32_t cols_p[NN]{};
    HIP_CHECK(hipMemcpy(cols_p, d_cols_p, NN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    const std::uint32_t wmma_p = host_sum16<Semantics::FPSan>(cols_p);

    // --- Report.
    std::printf("Device %s (warpSize=%d).\n", props.gcnArchName, wave);
    std::printf("Float-mode reductions (sum of 256 bf16-cast values, "
                "accumulated in f32):\n");
    show_float("CPU naive  (sequential)", cpu_f);
    show_float("wfred      (wave butterfly)", wfred_f);
    show_float("matrix bf16 (col-sums + host sum)", wmma_f);
    std::printf("\nFPSan-mode reductions (payload of the same sum):\n");
    show_payload("CPU naive  (sequential)", cpu_p);
    show_payload("wfred      (wave butterfly)", wfred_p);
    show_payload("matrix bf16 (col-sums + host sum)", wmma_p);

    // --- Invariants.
    bool float_disagree = !(cpu_f == wfred_f && wfred_f == wmma_f);
    bool fpsan_agree    = (cpu_p == wfred_p && wfred_p == wmma_p);
    std::printf("\nFloat results %s.\n",
                float_disagree ? "DISAGREE (expected: non-associative f32 add)"
                               : "all AGREE (unexpected for this input)");
    std::printf("FPSan results %s.\n",
                fpsan_agree ? "all AGREE (expected: payload ring is associative)"
                            : "DISAGREE (unexpected!)");

    (void)hipFree(d_in);
    (void)hipFree(d_wfred_f);
    (void)hipFree(d_cols_f);
    (void)hipFree(d_wfred_p);
    (void)hipFree(d_cols_p);

    if(!float_disagree)
        return 2;
    if(!fpsan_agree)
        return 3;
    std::printf("\nDemo OK: FPSan certifies the float differences are just the "
                "expected\nconsequences of non-associativity, not bugs.\n");
    return 0;
}
