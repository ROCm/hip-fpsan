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

    // Benchmark the width-sensitive scalar ops for one float type FT and one
    // semantics S. Templating on FT (not just S) is deliberate: the modular
    // arithmetic narrows per element width, so a regression can hide in a narrow
    // type while f32 looks fine -- exactly how an exp2/sqrt 128-bit regression
    // slipped through when only f32 was measured. The transcendental loops
    // (exp/log/sin/cos) exercise the powmod/rotor paths that arithmetic does not.
    template <class FT, fpsan::Semantics S>
    void bench_arith(std::vector<Result>& results, const char* wtag)
    {
        using V = fpsan::Value<FT, S, fpsan::Conversions::Explicit>;
        std::vector<V> a_values; // small positive (valid for sqrt/log)
        std::vector<V> b_values;
        std::vector<V> c_values;
        a_values.reserve(1024);
        b_values.reserve(1024);
        c_values.reserve(1024);
        for(int i = 0; i < 1024; ++i)
        {
            const float jitter = static_cast<float>((i * 37) % 997);
            a_values.emplace_back(sample<FT>(0.25f + jitter * 1.0e-4f)); // in (0, ~0.35]
            b_values.emplace_back(sample<FT>(0.99989f + jitter * 4.0e-8f));
            c_values.emplace_back(sample<FT>(0.00003f + jitter * 1.0e-8f));
        }

        auto add_mul = [a_values, b_values, c_values](std::size_t iters) -> std::uint64_t {
            V             x(sample<FT>(1.25f));
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
            V             x(sample<FT>(1.25f));
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
            V             x(sample<FT>(1.25f));
            std::uint64_t acc = 0;
            for(std::size_t i = 0; i < iters; ++i)
            {
                x = fpsan::sqrt(x + a_values[(i + 11) & 1023]);
                if((i & 255) == 0)
                    acc = mix(acc, storage_u64(x));
            }
            return acc ^ storage_u64(x);
        };
        // Transcendentals: apply to bounded array values (no accumulation into x,
        // so the native baseline stays finite). exp/log use the exp/dlog channels;
        // sin/cos use the rotor channel -- all powmod-based on the faithful rings.
        auto unary_loop = [a_values](auto op) {
            return [a_values, op](std::size_t iters) -> std::uint64_t {
                std::uint64_t acc = 0;
                for(std::size_t i = 0; i < iters; ++i)
                    acc = mix(acc, storage_u64(op(a_values[i & 1023])));
                return acc;
            };
        };
        auto exp_loop = unary_loop([](V v) { return fpsan::exp(v); });
        auto log_loop = unary_loop([](V v) { return fpsan::log(v); });
        auto sin_loop = unary_loop([](V v) { return fpsan::sin(v); });
        auto cos_loop = unary_loop([](V v) { return fpsan::cos(v); });

        const char* s = fpsan::semantics_name(S);
        auto        nm = [wtag](const char* op) { return std::string(wtag) + " " + op; };
        results.push_back(
            run_case("arithmetic", nm("add/mul/sub").c_str(), s, add_mul, 4096, 1u << 26, 70.0));
        results.push_back(
            run_case("arithmetic", nm("add/div").c_str(), s, div_loop, 1024, 1u << 24, 70.0));
        results.push_back(
            run_case("arithmetic", nm("add/sqrt").c_str(), s, sqrt_loop, 1024, 1u << 24, 70.0));
        results.push_back(
            run_case("transcendental", nm("exp").c_str(), s, exp_loop, 1024, 1u << 23, 70.0));
        results.push_back(
            run_case("transcendental", nm("log").c_str(), s, log_loop, 1024, 1u << 23, 70.0));
        results.push_back(
            run_case("transcendental", nm("sin").c_str(), s, sin_loop, 1024, 1u << 23, 70.0));
        results.push_back(
            run_case("transcendental", nm("cos").c_str(), s, cos_loop, 1024, 1u << 23, 70.0));
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

    // Run every semantics at both a 32-bit and a 16-bit element width: the
    // modular arithmetic narrows per width, so narrow types must be measured
    // directly (an f32-only suite hid a 128-bit regression in narrow widths).
#define FPSAN_BENCH_ARITH_ALL_SEMANTICS(FT, WTAG)                        \
    bench_arith<FT, fpsan::Semantics::Native>(results, WTAG);            \
    bench_arith<FT, fpsan::Semantics::Triton>(results, WTAG);            \
    bench_arith<FT, fpsan::Semantics::Field>(results, WTAG);             \
    bench_arith<FT, fpsan::Semantics::FieldFast>(results, WTAG);         \
    bench_arith<FT, fpsan::Semantics::FieldWithMulCasts>(results, WTAG); \
    bench_arith<FT, fpsan::Semantics::SophieGermainRing>(results, WTAG); \
    bench_arith<FT, fpsan::Semantics::PythagoreanRing>(results, WTAG);
    FPSAN_BENCH_ARITH_ALL_SEMANTICS(float, "f32")
    FPSAN_BENCH_ARITH_ALL_SEMANTICS(_Float16, "f16")
#undef FPSAN_BENCH_ARITH_ALL_SEMANTICS

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
