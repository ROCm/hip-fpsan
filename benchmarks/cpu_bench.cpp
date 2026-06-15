// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Host microbenchmarks for FPSan semantics. Results are CSV and normalized to
// Semantics::Native for each case.

#include <fpsan/fpsan.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

    volatile std::uint64_t g_sink = 0;

    struct Result
    {
        std::string   group;
        std::string   case_name;
        std::string   semantics;
        double        ns_per_iter = 0.0;
        std::size_t   iters       = 0;
        std::uint64_t checksum    = 0;
    };

    template <class T>
    T sample(float x)
    {
        if constexpr(std::is_same_v<T, fpsan::fp8_e4m3> || std::is_same_v<T, fpsan::fp8_e5m2>)
            return T(x);
        else
            return static_cast<T>(x);
    }

    template <class V>
    std::uint64_t storage_u64(const V& v)
    {
        return static_cast<std::uint64_t>(v.to_storage_bits());
    }

    std::uint64_t mix(std::uint64_t acc, std::uint64_t bits)
    {
        return acc ^ (bits + 0x9e3779b97f4a7c15ull + (acc << 6) + (acc >> 2));
    }

    template <class Fn>
    double time_once(Fn&& fn, std::size_t iters, std::uint64_t& checksum)
    {
        const auto start = std::chrono::steady_clock::now();
        checksum ^= fn(iters);
        const auto stop = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::nano>(stop - start).count()
               / static_cast<double>(iters);
    }

    template <class Fn>
    Result run_case(const char* group,
                    const char* case_name,
                    const char* semantics,
                    Fn&&        fn,
                    std::size_t min_iters,
                    std::size_t max_iters,
                    double      target_ms)
    {
        std::size_t   iters    = min_iters;
        std::uint64_t checksum = 0;
        for(;;)
        {
            const double ns_per_iter = time_once(fn, iters, checksum);
            const double elapsed_ms  = ns_per_iter * static_cast<double>(iters) / 1.0e6;
            if(elapsed_ms >= target_ms || iters >= max_iters)
                break;
            iters = std::min<std::size_t>(iters * 2, max_iters);
        }

        double best_ns = std::numeric_limits<double>::infinity();
        for(int repeat = 0; repeat < 4; ++repeat)
            best_ns = std::min(best_ns, time_once(fn, iters, checksum));

        g_sink ^= checksum;
        return Result{group, case_name, semantics, best_ns, iters, checksum};
    }

    template <fpsan::Semantics S>
    void bench_arith(std::vector<Result>& results)
    {
        using V = fpsan::Value<float, S, fpsan::Conversions::Explicit>;
        std::vector<V> a_values;
        std::vector<V> b_values;
        std::vector<V> c_values;
        a_values.reserve(1024);
        b_values.reserve(1024);
        c_values.reserve(1024);
        for(int i = 0; i < 1024; ++i)
        {
            const float jitter = static_cast<float>((i * 37) % 997);
            a_values.emplace_back(sample<float>(0.00011f + jitter * 1.0e-8f));
            b_values.emplace_back(sample<float>(0.99989f + jitter * 4.0e-8f));
            c_values.emplace_back(sample<float>(0.00003f + jitter * 1.0e-8f));
        }

        auto add_mul = [a_values, b_values, c_values](std::size_t iters) -> std::uint64_t {
            V             x(sample<float>(1.25f));
            std::uint64_t acc = 0;
            for(std::size_t i = 0; i < iters; ++i)
            {
                x = (x + a_values[i & 1023]) * b_values[(i + 17) & 1023]
                    - c_values[(i + 73) & 1023];
                if((i & 255) == 0)
                    acc = mix(acc, storage_u64(x));
            }
            return acc ^ storage_u64(x);
        };
        auto div_loop = [a_values, b_values](std::size_t iters) -> std::uint64_t {
            V             x(sample<float>(1.25f));
            std::uint64_t acc = 0;
            for(std::size_t i = 0; i < iters; ++i)
            {
                x = (x + a_values[(i + 5) & 1023]) / b_values[(i + 97) & 1023];
                if((i & 255) == 0)
                    acc = mix(acc, storage_u64(x));
            }
            return acc ^ storage_u64(x);
        };
        auto sqrt_loop = [a_values](std::size_t iters) -> std::uint64_t {
            V             x(sample<float>(1.25f));
            std::uint64_t acc = 0;
            for(std::size_t i = 0; i < iters; ++i)
            {
                x = fpsan::sqrt(x + a_values[(i + 11) & 1023]);
                if((i & 255) == 0)
                    acc = mix(acc, storage_u64(x));
            }
            return acc ^ storage_u64(x);
        };

        const char* s = fpsan::semantics_name(S);
        results.push_back(
            run_case("arithmetic", "f32 add/mul/sub", s, add_mul, 4096, 1u << 26, 70.0));
        results.push_back(run_case("arithmetic", "f32 add/div", s, div_loop, 1024, 1u << 24, 70.0));
        results.push_back(
            run_case("arithmetic", "f32 add/sqrt", s, sqrt_loop, 1024, 1u << 24, 70.0));
    }

    template <fpsan::Semantics S, class FromFT, class ToFT>
    void bench_cast(std::vector<Result>& results,
                    const char*          case_name,
                    std::size_t          min_iters,
                    std::size_t          max_iters,
                    double               target_ms)
    {
        using From = fpsan::Value<FromFT, S, fpsan::Conversions::Explicit>;
        std::vector<From> inputs;
        inputs.reserve(1024);
        for(int i = 0; i < 1024; ++i)
        {
            const float x = 0.125f + static_cast<float>((i * 37) % 997) * 0.00091f;
            inputs.emplace_back(sample<FromFT>(x));
        }

        auto fn = [inputs = std::move(inputs)](std::size_t iters) -> std::uint64_t {
            std::uint64_t acc = 0;
            for(std::size_t i = 0; i < iters; ++i)
            {
                const auto y = fpsan::cast<ToFT>(inputs[i & 1023]);
                acc          = mix(acc, storage_u64(y));
            }
            return acc;
        };

        results.push_back(run_case(
            "cast", case_name, fpsan::semantics_name(S), fn, min_iters, max_iters, target_ms));
    }

    template <fpsan::Semantics S>
    void bench_casts_for_semantics(std::vector<Result>& results)
    {
        constexpr bool expensive = fpsan::detail::has_multiplicative_field_casts(S);
        bench_cast<S, fpsan::fp8_e4m3, float>(
            results, "fp8->f32", 1024, expensive ? 1u << 20 : 1u << 25, 60.0);
        bench_cast<S, float, fpsan::fp8_e4m3>(
            results, "f32->fp8", 1024, expensive ? 1u << 20 : 1u << 25, 60.0);
        bench_cast<S, _Float16, float>(
            results, "f16->f32", 1024, expensive ? 1u << 20 : 1u << 25, 60.0);
        bench_cast<S, float, _Float16>(
            results, "f32->f16", 1024, expensive ? 1u << 20 : 1u << 25, 60.0);
        bench_cast<S, float, double>(
            results, "f32->f64", 16, expensive ? 1u << 17 : 1u << 25, 60.0);
        bench_cast<S, double, float>(
            results, "f64->f32", 16, expensive ? 1u << 17 : 1u << 25, 60.0);
    }

    void print_results(const std::vector<Result>& results)
    {
        std::map<std::pair<std::string, std::string>, double> native_baseline;
        for(const Result& r : results)
        {
            if(r.semantics == "Native")
                native_baseline[{r.group, r.case_name}] = r.ns_per_iter;
        }

        std::printf("backend,group,case,semantics,ns_per_iter,native_x,iters,checksum\n");
        for(const Result& r : results)
        {
            const auto   key      = std::make_pair(r.group, r.case_name);
            const double baseline = native_baseline.at(key);
            std::printf("CPU,%s,%s,%s,%.6f,%.6f,%zu,%016llx\n",
                        r.group.c_str(),
                        r.case_name.c_str(),
                        r.semantics.c_str(),
                        r.ns_per_iter,
                        r.ns_per_iter / baseline,
                        r.iters,
                        static_cast<unsigned long long>(r.checksum));
        }
        std::printf("sink,,,,,,,%016llx\n", static_cast<unsigned long long>(g_sink));
    }

} // namespace

int main()
{
    std::vector<Result> results;
    results.reserve(80);

    bench_arith<fpsan::Semantics::Native>(results);
    bench_arith<fpsan::Semantics::Triton>(results);
    bench_arith<fpsan::Semantics::Field>(results);
    bench_arith<fpsan::Semantics::FieldFast>(results);
    bench_arith<fpsan::Semantics::FieldWithMulCasts>(results);
    bench_arith<fpsan::Semantics::SophieGermainRing>(results);
    bench_arith<fpsan::Semantics::PythagoreanRing>(results);

    bench_casts_for_semantics<fpsan::Semantics::Native>(results);
    bench_casts_for_semantics<fpsan::Semantics::Triton>(results);
    bench_casts_for_semantics<fpsan::Semantics::Field>(results);
    bench_casts_for_semantics<fpsan::Semantics::FieldFast>(results);
    bench_casts_for_semantics<fpsan::Semantics::FieldWithMulCasts>(results);
    bench_casts_for_semantics<fpsan::Semantics::SophieGermainRing>(results);
    bench_casts_for_semantics<fpsan::Semantics::PythagoreanRing>(results);

    print_results(results);
    return 0;
}
