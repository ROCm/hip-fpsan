// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/parity_ops.hpp
// ----------------------------------------------------------------------------
// Single source of truth for "what the core computes" from one (a, b, c) input
// triple. parity_compute() is FPSAN_HOST_DEVICE and is instantiated ONCE; the
// parity test runs it both on the host and (in a kernel) on the device and
// asserts the two agree bit-for-bit. This lifts the host<->device alignment of
// the core algebra from a compiled-but-unexecuted property to a runtime-checked
// one, without maintaining separate host and device test bodies.
//
// All observables are extracted via Value::to_storage_bits(), which returns the
// raw stored representation in BOTH modes (the integer payload in FPSan mode,
// the float's bits in Float mode) -- so a single code path covers every config.
//
// Scope of the host<->device guarantee: in FPSan mode every result is integer /
// constexpr payload algebra and must be bit-identical on host and device. Float
// mode is a passthrough to native hardware arithmetic, whose rounding can
// legitimately differ between a host (often emulated, esp. for _Float16/__bf16)
// and the GPU; bit-exact host/device conformance of native FP is NOT a property
// this suite asserts. The test therefore compares all observables for FPSan mode
// and only the rounding-independent ones (embed round-trip, negation, the b==0
// flag) for Float mode.
// ----------------------------------------------------------------------------
#ifndef FPSAN_TESTS_PARITY_OPS_HPP
#define FPSAN_TESTS_PARITY_OPS_HPP

#include "fpsan/fpsan.hpp"

#include <cstdint>

namespace fpsan_test
{

    // Algebraic-law check bits (set when the law held for this input triple).
    enum Law : std::uint32_t
    {
        kAssocAdd = 1u << 0, // (a+b)+c == a+(b+c)
        kAssocMul = 1u << 1, // (a*b)*c == a*(b*c)
        kDistrib  = 1u << 2, // a*(b+c) == a*b + a*c
        kCommAdd  = 1u << 3, // a+b == b+a
        kCommMul  = 1u << 4, // a*b == b*a
        kLess     = 1u << 5, // a < b
        kEqual    = 1u << 6, // a == b
    };

    // The exact ring laws (hold in FPSan mode for every finite input).
    inline constexpr std::uint32_t kRingLaws
        = kAssocAdd | kAssocMul | kDistrib | kCommAdd | kCommMul;

    // A bundle of bit-exact observables. Trivially copyable so it crosses the
    // host<->device boundary verbatim.
    struct ParitySample
    {
        std::uint64_t storage_a; // bits of the stored repr of Value(a)
        std::uint64_t storage_add; // ... of (A + B)
        std::uint64_t storage_sub; // ... of (A - B)
        std::uint64_t storage_mul; // ... of (A * B)
        std::uint64_t storage_div; // ... of (A / B), or 0 when b == 0
        std::uint64_t storage_neg; // ... of (-A)
        std::uint32_t laws; // bitmask of Law
        std::uint32_t b_is_zero; // 1 when the division was skipped
    };

    template <class FT, fpsan::Semantics S, fpsan::Conversions Cv>
    FPSAN_HOST_DEVICE inline ParitySample parity_compute(FT a, FT b, FT c)
    {
        using F = fpsan::Value<FT, S, Cv>;
        F A(a), B(b), C(c);

        ParitySample r{};
        r.storage_a   = static_cast<std::uint64_t>(A.to_storage_bits());
        r.storage_add = static_cast<std::uint64_t>((A + B).to_storage_bits());
        r.storage_sub = static_cast<std::uint64_t>((A - B).to_storage_bits());
        r.storage_mul = static_cast<std::uint64_t>((A * B).to_storage_bits());
        r.storage_neg = static_cast<std::uint64_t>((-A).to_storage_bits());

        const bool bzero = (b == FT(0));
        r.b_is_zero      = bzero ? 1u : 0u;
        r.storage_div
            = bzero ? std::uint64_t{0} : static_cast<std::uint64_t>((A / B).to_storage_bits());

        std::uint32_t laws = 0;
        if((A + B) + C == A + (B + C))
            laws |= kAssocAdd;
        if((A * B) * C == A * (B * C))
            laws |= kAssocMul;
        if(A * (B + C) == A * B + A * C)
            laws |= kDistrib;
        if(A + B == B + A)
            laws |= kCommAdd;
        if(A * B == B * A)
            laws |= kCommMul;
        if(A < B)
            laws |= kLess;
        if(A == B)
            laws |= kEqual;
        r.laws = laws;
        return r;
    }

} // namespace fpsan_test

#endif // FPSAN_TESTS_PARITY_OPS_HPP
