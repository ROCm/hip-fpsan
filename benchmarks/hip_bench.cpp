// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// HIP microbenchmarks for FPSan semantics. Timing uses hipEvent elapsed time
// around kernel launches, so the reported time is device-side execution time.
// Arithmetic inputs are pre-embedded on the host and loaded with from_storage_bits()
// inside kernels; this measures arithmetic itself, not repeated embedding.

#include <hip/hip_runtime.h>

#include <fpsan/fpsan.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define HIP_CHECK(expr)                                                                          \
    do                                                                                           \
    {                                                                                            \
        hipError_t err__ = (expr);                                                               \
        if(err__ != hipSuccess)                                                                  \
        {                                                                                        \
            std::fprintf(                                                                        \
                stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(err__)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while(false)

namespace
{

    struct Result
    {
        std::string        group;
        std::string        case_name;
        std::string        semantics;
        double             ns_per_iter = 0.0;
        int                threads     = 0;
        int                iters       = 0;
        unsigned long long checksum    = 0;
    };

    template <class T>
    FPSAN_HOST_DEVICE T sample(float x)
    {
        if constexpr(std::is_same_v<T, fpsan::fp8_e4m3> || std::is_same_v<T, fpsan::fp8_e5m2>)
            return T(x);
        else
            return static_cast<T>(x);
    }

    template <class V>
    FPSAN_HOST_DEVICE unsigned long long storage_u64(const V& v)
    {
        return static_cast<unsigned long long>(v.to_storage_bits());
    }

    FPSAN_HOST_DEVICE unsigned long long mix(unsigned long long acc, unsigned long long bits)
    {
        return acc ^ (bits + 0x9e3779b97f4a7c15ull + (acc << 6) + (acc >> 2));
    }

    template <class FT, fpsan::Semantics S>
    __global__ void addmul_preembed_kernel(
        const typename fpsan::Value<FT, S, fpsan::Conversions::Explicit>::bits_type* a_in,
        const typename fpsan::Value<FT, S, fpsan::Conversions::Explicit>::bits_type* b_in,
        const typename fpsan::Value<FT, S, fpsan::Conversions::Explicit>::bits_type* c_in,
        unsigned                                                                     input_mask,
        unsigned long long*                                                          out,
        int                                                                          iters)
    {
        using V                = fpsan::Value<FT, S, fpsan::Conversions::Explicit>;
        const int          tid = blockIdx.x * blockDim.x + threadIdx.x;
        V                  x = V::from_storage_bits(a_in[static_cast<unsigned>(tid) & input_mask]);
        unsigned long long checksum = 0;
        for(int i = 0; i < iters; ++i)
        {
            const unsigned ia
                = (static_cast<unsigned>(tid) + 17u * static_cast<unsigned>(i)) & input_mask;
            const unsigned ib
                = (static_cast<unsigned>(tid) + 29u * static_cast<unsigned>(i)) & input_mask;
            const unsigned ic
                = (static_cast<unsigned>(tid) + 43u * static_cast<unsigned>(i)) & input_mask;
            const V a = V::from_storage_bits(a_in[ia]);
            const V b = V::from_storage_bits(b_in[ib]);
            const V c = V::from_storage_bits(c_in[ic]);
            x         = (x + a) * b - c;
            checksum  = mix(checksum, storage_u64(x));
        }
        out[tid] = checksum ^ storage_u64(x);
    }

    template <class FT, fpsan::Semantics S>
    __global__ void div_preembed_kernel(
        const typename fpsan::Value<FT, S, fpsan::Conversions::Explicit>::bits_type* a_in,
        const typename fpsan::Value<FT, S, fpsan::Conversions::Explicit>::bits_type* b_in,
        unsigned                                                                     input_mask,
        unsigned long long*                                                          out,
        int                                                                          iters)
    {
        using V                = fpsan::Value<FT, S, fpsan::Conversions::Explicit>;
        const int          tid = blockIdx.x * blockDim.x + threadIdx.x;
        V                  x = V::from_storage_bits(a_in[static_cast<unsigned>(tid) & input_mask]);
        unsigned long long checksum = 0;
        for(int i = 0; i < iters; ++i)
        {
            const unsigned ia
                = (static_cast<unsigned>(tid) + 17u * static_cast<unsigned>(i)) & input_mask;
            const unsigned ib
                = (static_cast<unsigned>(tid) + 29u * static_cast<unsigned>(i)) & input_mask;
            const V a = V::from_storage_bits(a_in[ia]);
            const V b = V::from_storage_bits(b_in[ib]);
            x         = (x + a) / b;
            checksum  = mix(checksum, storage_u64(x));
        }
        out[tid] = checksum ^ storage_u64(x);
    }

    template <class FT, fpsan::Semantics S>
    __global__ void sqrt_preembed_kernel(
        const typename fpsan::Value<FT, S, fpsan::Conversions::Explicit>::bits_type* a_in,
        unsigned                                                                     input_mask,
        unsigned long long*                                                          out,
        int                                                                          iters)
    {
        using V                = fpsan::Value<FT, S, fpsan::Conversions::Explicit>;
        const int          tid = blockIdx.x * blockDim.x + threadIdx.x;
        V                  x = V::from_storage_bits(a_in[static_cast<unsigned>(tid) & input_mask]);
        unsigned long long checksum = 0;
        for(int i = 0; i < iters; ++i)
        {
            const unsigned ia
                = (static_cast<unsigned>(tid) + 17u * static_cast<unsigned>(i)) & input_mask;
            const V a = V::from_storage_bits(a_in[ia]);
            x         = fpsan::sqrt(x + a);
            checksum  = mix(checksum, storage_u64(x));
        }
        out[tid] = checksum ^ storage_u64(x);
    }

    // Generic unary transcendental kernel: applies Op to bounded array inputs
    // (no accumulation, so the native baseline stays finite). Exercises the
    // powmod (exp/log) and rotor (sin/cos) channels that arithmetic skips -- the
    // paths where a per-width 128-bit regression is catastrophic on GPU.
    struct OpExp
    {
        template <class V>
        __device__ static V apply(V v)
        {
            return fpsan::exp(v);
        }
    };
    struct OpLog
    {
        template <class V>
        __device__ static V apply(V v)
        {
            return fpsan::log(v);
        }
    };
    struct OpSin
    {
        template <class V>
        __device__ static V apply(V v)
        {
            return fpsan::sin(v);
        }
    };
    struct OpCos
    {
        template <class V>
        __device__ static V apply(V v)
        {
            return fpsan::cos(v);
        }
    };
    template <class FT, fpsan::Semantics S, class Op>
    __global__ void unary_preembed_kernel(
        const typename fpsan::Value<FT, S, fpsan::Conversions::Explicit>::bits_type* a_in,
        unsigned                                                                     input_mask,
        unsigned long long*                                                          out,
        int                                                                          iters)
    {
        using V                     = fpsan::Value<FT, S, fpsan::Conversions::Explicit>;
        const int          tid      = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long long checksum = 0;
        for(int i = 0; i < iters; ++i)
        {
            const unsigned ia
                = (static_cast<unsigned>(tid) + 17u * static_cast<unsigned>(i)) & input_mask;
            const V a = V::from_storage_bits(a_in[ia]);
            checksum  = mix(checksum, storage_u64(Op::apply(a)));
        }
        out[tid] = checksum;
    }

    template <fpsan::Semantics S, class FromFT, class ToFT>
    __global__ void cast_kernel(
        const typename fpsan::Value<FromFT, S, fpsan::Conversions::Explicit>::bits_type* in,
        unsigned                                                                         input_mask,
        unsigned long long*                                                              out,
        int                                                                              iters)
    {
        using From                  = fpsan::Value<FromFT, S, fpsan::Conversions::Explicit>;
        const int          tid      = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long long checksum = 0;
        for(int i = 0; i < iters; ++i)
        {
            const unsigned idx
                = (static_cast<unsigned>(tid) + 33u * static_cast<unsigned>(i)) & input_mask;
            const auto x = From::from_storage_bits(in[idx]);
            const auto y = fpsan::cast<ToFT>(x);
            checksum     = mix(checksum, storage_u64(y));
        }
        out[tid] = checksum;
    }

    unsigned long long checksum_device_output(unsigned long long* d_out, int count)
    {
        std::vector<unsigned long long> h_out(count);
        HIP_CHECK(hipMemcpy(
            h_out.data(), d_out, h_out.size() * sizeof(unsigned long long), hipMemcpyDeviceToHost));
        unsigned long long checksum = 0;
        for(unsigned long long x : h_out)
            checksum = mix(checksum, x);
        return checksum;
    }

    template <class Launch>
    Result run_gpu_case(const char* group,
                        const char* case_name,
                        const char* semantics,
                        int         blocks,
                        int         threads_per_block,
                        int         iters,
                        Launch&&    launch)
    {
        const int           count = blocks * threads_per_block;
        unsigned long long* d_out = nullptr;
        HIP_CHECK(hipMalloc(&d_out, static_cast<std::size_t>(count) * sizeof(unsigned long long)));

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        launch(d_out);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        float best_ms = std::numeric_limits<float>::infinity();
        for(int repeat = 0; repeat < 5; ++repeat)
        {
            HIP_CHECK(hipEventRecord(start));
            launch(d_out);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
            best_ms = std::min(best_ms, ms);
        }

        const unsigned long long checksum = checksum_device_output(d_out, count);
        const double iterations           = static_cast<double>(count) * static_cast<double>(iters);
        const double ns_per_iter          = static_cast<double>(best_ms) * 1.0e6 / iterations;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_out));

        return Result{group, case_name, semantics, ns_per_iter, count, iters, checksum};
    }

    template <class FT, fpsan::Semantics S>
    void run_arith_for_semantics(std::vector<Result>& results, const char* wtag)
    {
        using V    = fpsan::Value<FT, S, fpsan::Conversions::Explicit>;
        using Bits = typename V::bits_type;

        constexpr unsigned input_count = 4096;
        std::vector<Bits>  h_a(input_count), h_b(input_count), h_c(input_count);
        for(unsigned i = 0; i < input_count; ++i)
        {
            const float jitter = static_cast<float>((i * 37) % 997);
            h_a[i]             = V(sample<FT>(0.25f + jitter * 1.0e-4f))
                         .to_storage_bits(); // (0,~0.35], ok sqrt/log
            h_b[i] = V(sample<FT>(0.99989f + jitter * 4.0e-8f)).to_storage_bits();
            h_c[i] = V(sample<FT>(0.00003f + jitter * 1.0e-8f)).to_storage_bits();
        }

        Bits* d_a = nullptr;
        Bits* d_b = nullptr;
        Bits* d_c = nullptr;
        HIP_CHECK(hipMalloc(&d_a, h_a.size() * sizeof(Bits)));
        HIP_CHECK(hipMalloc(&d_b, h_b.size() * sizeof(Bits)));
        HIP_CHECK(hipMalloc(&d_c, h_c.size() * sizeof(Bits)));
        HIP_CHECK(hipMemcpy(d_a, h_a.data(), h_a.size() * sizeof(Bits), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, h_b.data(), h_b.size() * sizeof(Bits), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_c, h_c.data(), h_c.size() * sizeof(Bits), hipMemcpyHostToDevice));

        const char* s  = fpsan::semantics_name(S);
        auto        nm = [wtag](const char* op) { return std::string(wtag) + " " + op; };
        results.push_back(run_gpu_case("arithmetic",
                                       nm("add/mul/sub").c_str(),
                                       s,
                                       256,
                                       256,
                                       1024,
                                       [=](unsigned long long* out) {
                                           addmul_preembed_kernel<FT, S><<<256, 256>>>(
                                               d_a, d_b, d_c, input_count - 1, out, 1024);
                                       }));
        results.push_back(run_gpu_case(
            "arithmetic", nm("add/div").c_str(), s, 256, 256, 512, [=](unsigned long long* out) {
                div_preembed_kernel<FT, S><<<256, 256>>>(d_a, d_b, input_count - 1, out, 512);
            }));
        results.push_back(run_gpu_case(
            "arithmetic", nm("add/sqrt").c_str(), s, 256, 256, 512, [=](unsigned long long* out) {
                sqrt_preembed_kernel<FT, S><<<256, 256>>>(d_a, input_count - 1, out, 512);
            }));
        // exp/sin/cos exercise the powmod (exp) and rotor (sin/cos) channels --
        // the bounded ~O(log n) loops where the 128-bit width regression lived.
        // log is deliberately omitted on GPU: faithful log is a discrete-log
        // SEARCH (Pollard rho), not a width-sensitive modexp, and its cost trips
        // the GPU watchdog. It is measured on the host (cpu_bench) instead.
        results.push_back(run_gpu_case(
            "transcendental", nm("exp").c_str(), s, 256, 256, 64, [=](unsigned long long* out) {
                unary_preembed_kernel<FT, S, OpExp><<<256, 256>>>(d_a, input_count - 1, out, 64);
            }));
        results.push_back(run_gpu_case(
            "transcendental", nm("sin").c_str(), s, 256, 256, 64, [=](unsigned long long* out) {
                unary_preembed_kernel<FT, S, OpSin><<<256, 256>>>(d_a, input_count - 1, out, 64);
            }));
        results.push_back(run_gpu_case(
            "transcendental", nm("cos").c_str(), s, 256, 256, 64, [=](unsigned long long* out) {
                unary_preembed_kernel<FT, S, OpCos><<<256, 256>>>(d_a, input_count - 1, out, 64);
            }));

        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
    }

    template <fpsan::Semantics S, class FromFT, class ToFT>
    void run_cast_case(std::vector<Result>& results,
                       const char*          case_name,
                       int                  blocks,
                       int                  threads_per_block,
                       int                  iters)
    {
        using From = fpsan::Value<FromFT, S, fpsan::Conversions::Explicit>;
        using Bits = typename From::bits_type;

        constexpr unsigned input_count = 4096;
        std::vector<Bits>  h_in(input_count);
        for(unsigned i = 0; i < input_count; ++i)
        {
            const float x = 0.125f + static_cast<float>((i * 37) % 997) * 0.00091f;
            h_in[i]       = From(sample<FromFT>(x)).to_storage_bits();
        }

        Bits* d_in = nullptr;
        HIP_CHECK(hipMalloc(&d_in, h_in.size() * sizeof(Bits)));
        HIP_CHECK(hipMemcpy(d_in, h_in.data(), h_in.size() * sizeof(Bits), hipMemcpyHostToDevice));

        results.push_back(
            run_gpu_case("cast",
                         case_name,
                         fpsan::semantics_name(S),
                         blocks,
                         threads_per_block,
                         iters,
                         [d_in, blocks, threads_per_block, iters](unsigned long long* out) {
                             cast_kernel<S, FromFT, ToFT>
                                 <<<blocks, threads_per_block>>>(d_in, input_count - 1, out, iters);
                         }));

        HIP_CHECK(hipFree(d_in));
    }

    template <fpsan::Semantics S>
    void run_casts_for_semantics(std::vector<Result>& results)
    {
        constexpr bool expensive = fpsan::detail::has_multiplicative_field_casts(S);
        const int      blocks    = expensive ? 128 : 256;
        const int      iters     = expensive ? 8 : 256;
        run_cast_case<S, fpsan::fp8_e4m3, float>(results, "fp8->f32", blocks, 256, iters);
        run_cast_case<S, float, fpsan::fp8_e4m3>(results, "f32->fp8", blocks, 256, iters);
        run_cast_case<S, _Float16, float>(results, "f16->f32", blocks, 256, iters);
        run_cast_case<S, float, _Float16>(results, "f32->f16", blocks, 256, iters);
        run_cast_case<S, float, double>(
            results, "f32->f64", expensive ? 16 : 256, 256, expensive ? 1 : 256);
        run_cast_case<S, double, float>(
            results, "f64->f32", expensive ? 16 : 256, 256, expensive ? 1 : 256);
    }

    void print_results(const char*                device_name,
                       const char*                arch_name,
                       const std::vector<Result>& results)
    {
        std::map<std::pair<std::string, std::string>, double> native_baseline;
        for(const Result& r : results)
        {
            if(r.semantics == "Native")
                native_baseline[{r.group, r.case_name}] = r.ns_per_iter;
        }

        std::printf("device,%s,%s\n", device_name, arch_name);
        std::printf("backend,group,case,semantics,ns_per_iter,native_x,threads,iters,checksum\n");
        for(const Result& r : results)
        {
            const auto   key      = std::make_pair(r.group, r.case_name);
            const double baseline = native_baseline.at(key);
            std::printf("HIP,%s,%s,%s,%.6f,%.6f,%d,%d,%016llx\n",
                        r.group.c_str(),
                        r.case_name.c_str(),
                        r.semantics.c_str(),
                        r.ns_per_iter,
                        r.ns_per_iter / baseline,
                        r.threads,
                        r.iters,
                        r.checksum);
        }
    }

} // namespace

int main()
{
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if(device_count == 0)
    {
        std::fprintf(stderr, "No HIP devices found\n");
        return 1;
    }

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::vector<Result> results;
    results.reserve(80);

    // Both a 32-bit and a 16-bit element width: the modular arithmetic narrows
    // per width, so narrow types must be measured directly on the GPU (an
    // f32-only suite hid a 128-bit regression in narrow widths and transcendentals).
#define FPSAN_GPU_ARITH_ALL_SEMANTICS(FT, WTAG)                                      \
    run_arith_for_semantics<FT, fpsan::Semantics::Native>(results, WTAG);            \
    run_arith_for_semantics<FT, fpsan::Semantics::Triton>(results, WTAG);            \
    run_arith_for_semantics<FT, fpsan::Semantics::Field>(results, WTAG);             \
    run_arith_for_semantics<FT, fpsan::Semantics::FieldFast>(results, WTAG);         \
    run_arith_for_semantics<FT, fpsan::Semantics::FieldWithMulCasts>(results, WTAG); \
    run_arith_for_semantics<FT, fpsan::Semantics::SophieGermainRing>(results, WTAG); \
    run_arith_for_semantics<FT, fpsan::Semantics::PythagoreanRing>(results, WTAG);
    FPSAN_GPU_ARITH_ALL_SEMANTICS(float, "f32")
    FPSAN_GPU_ARITH_ALL_SEMANTICS(_Float16, "f16")
#undef FPSAN_GPU_ARITH_ALL_SEMANTICS

    run_casts_for_semantics<fpsan::Semantics::Native>(results);
    run_casts_for_semantics<fpsan::Semantics::Triton>(results);
    run_casts_for_semantics<fpsan::Semantics::Field>(results);
    run_casts_for_semantics<fpsan::Semantics::FieldFast>(results);
    run_casts_for_semantics<fpsan::Semantics::FieldWithMulCasts>(results);
    run_casts_for_semantics<fpsan::Semantics::SophieGermainRing>(results);
    run_casts_for_semantics<fpsan::Semantics::PythagoreanRing>(results);

    print_results(prop.name, prop.gcnArchName, results);
    return 0;
}
