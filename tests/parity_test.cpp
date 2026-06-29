// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/parity_test.cpp
// ----------------------------------------------------------------------------
// Host<->device parity for the core Value algebra, from a single source body
// (parity_ops.hpp). The same parity_compute() is run on the host and -- when
// compiled as HIP with a device present -- inside a kernel; the two result
// bundles must be bit-for-bit identical. Built in BOTH the pure-C++ and HIP
// configs:
//   * pure C++ : compiles the shared body as ordinary C++ and checks the host
//                invariants (exact ring laws in FPSan mode, commutativity in
//                both). Fast, no GPU.
//   * HIP      : additionally executes the body on the device and asserts
//                device == host for every observable.
// ----------------------------------------------------------------------------
#include "parity_ops.hpp"

#include "test_random.hpp"

#include <gtest/gtest.h>

#include <vector>

#if defined(__HIP__)
#include "hip_test_utils.hpp"
#endif

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan_test::ParitySample;

namespace {

// Deterministic inputs: a few exact-integer anchors (exercise 0/1/-1 fixed
// points and a b==0 division-skip) followed by pseudo-random quarters.
template <class FT>
void fill_inputs(std::vector<FT> &a, std::vector<FT> &b, std::vector<FT> &c, int n) {
  const FT anchors[] = {FT(0), FT(1), FT(-1), FT(2), FT(-2)};
  std::mt19937 rng = fpsan_test::make_rng();
  for (int i = 0; i < n; ++i) {
    if (i < 5) {
      a.push_back(anchors[i]);
      b.push_back(anchors[(i + 1) % 5]); // b==0 at i==4 -> division skipped
      c.push_back(anchors[(i + 2) % 5]);
    } else {
      a.push_back(fpsan_test::pick_quarter<FT>(rng, -36, 36)); // -9 .. 9
      b.push_back(fpsan_test::pick_quarter<FT>(rng, -36, 36));
      c.push_back(fpsan_test::pick_quarter<FT>(rng, -36, 36));
    }
  }
}

#if defined(__HIP__)
template <class FT, Semantics S, Conversions Cv>
__global__ void parity_kernel(const FT *a, const FT *b, const FT *c, ParitySample *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  out[i] = fpsan_test::parity_compute<FT, S, Cv>(a[i], b[i], c[i]);
}

// Run the kernel and assert every observable equals the host result.
template <class FT, Semantics S, Conversions Cv>
void check_device_matches_host(const std::vector<FT> &a, const std::vector<FT> &b,
                               const std::vector<FT> &c, const std::vector<ParitySample> &host,
                               const char *label) {
  int ndev = 0;
  hipError_t e = hipGetDeviceCount(&ndev);
  if (e != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device available";

  const int n = static_cast<int>(a.size());
  FT *da, *db, *dc;
  ParitySample *dout;
  HIP_CHECK(hipMalloc(&da, n * sizeof(FT)));
  HIP_CHECK(hipMalloc(&db, n * sizeof(FT)));
  HIP_CHECK(hipMalloc(&dc, n * sizeof(FT)));
  HIP_CHECK(hipMalloc(&dout, n * sizeof(ParitySample)));
  HIP_CHECK(hipMemcpy(da, a.data(), n * sizeof(FT), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(db, b.data(), n * sizeof(FT), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dc, c.data(), n * sizeof(FT), hipMemcpyHostToDevice));

  parity_kernel<FT, S, Cv><<<dim3((n + 63) / 64), dim3(64)>>>(da, db, dc, dout, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<ParitySample> dev(n);
  HIP_CHECK(hipMemcpy(dev.data(), dout, n * sizeof(ParitySample), hipMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    const ParitySample &h = host[i];
    const ParitySample &d = dev[i];
    // Rounding-independent in both modes: embed round-trip, negation
    // (a sign-bit flip), and the division-skip flag.
    EXPECT_EQ(d.storage_a, h.storage_a) << label << " storage_a @" << i;
    EXPECT_EQ(d.storage_neg, h.storage_neg) << label << " neg @" << i;
    EXPECT_EQ(d.b_is_zero, h.b_is_zero) << label << " b_is_zero @" << i;
    // Every FPSan-family mode (Triton AND the algebraic
    // algebraic models) is integer/constexpr payload algebra: it MUST match
    // bit-for-bit. Native mode is native hardware arithmetic, whose
    // rounding may differ host vs device -- out of scope here.
    if constexpr (S != Semantics::Native) {
      EXPECT_EQ(d.storage_add, h.storage_add) << label << " add @" << i;
      EXPECT_EQ(d.storage_sub, h.storage_sub) << label << " sub @" << i;
      EXPECT_EQ(d.storage_mul, h.storage_mul) << label << " mul @" << i;
      EXPECT_EQ(d.storage_div, h.storage_div) << label << " div @" << i;
      EXPECT_EQ(d.laws, h.laws) << label << " laws @" << i;
    }
  }

  (void)hipFree(da);
  (void)hipFree(db);
  (void)hipFree(dc);
  (void)hipFree(dout);
}
#endif // __HIP__

template <class FT, Semantics S, Conversions Cv> void run_parity(const char *label) {
  const int n = 64;
  std::vector<FT> a, b, c;
  a.reserve(n);
  b.reserve(n);
  c.reserve(n);
  fill_inputs<FT>(a, b, c, n);

  std::vector<ParitySample> host(n);
  for (int i = 0; i < n; ++i)
    host[i] = fpsan_test::parity_compute<FT, S, Cv>(a[i], b[i], c[i]);

  // Host invariant on the shared body: in every FPSan-family mode (Triton
  // and the algebraic models) the payload ring laws are exact and
  // must always hold. Native mode is native arithmetic, so no algebraic-law
  // conformance is asserted.
  if constexpr (S != Semantics::Native)
    for (int i = 0; i < n; ++i)
      EXPECT_EQ(host[i].laws & fpsan_test::kRingLaws, fpsan_test::kRingLaws)
          << label << " ring law @" << i;

#if defined(__HIP__)
  check_device_matches_host<FT, S, Cv>(a, b, c, host, label);
#endif
}

} // namespace

#define PARITY_ONE(FT, TAG, SEM, CONV)                                                             \
  TEST(Parity, TAG##_##SEM##_##CONV) {                                                             \
    run_parity<FT, Semantics::SEM, Conversions::CONV>(#TAG "/" #SEM "/" #CONV);                    \
  }

#define PARITY_ALL_MODES(FT, TAG)                                                                  \
  PARITY_ONE(FT, TAG, Native, Implicit)                                                            \
  PARITY_ONE(FT, TAG, Native, Explicit)                                                            \
  PARITY_ONE(FT, TAG, Triton, Implicit)                                                            \
  PARITY_ONE(FT, TAG, Triton, Explicit)

// The default algebraic models. Conversions is orthogonal to the payload algebra
// here, so one representative (Explicit) per semantics suffices. The 2-suffixed
// reroll variants are covered by the core algebraic tests rather than this
// host/device parity smoke test.
#define PARITY_ALG_MODES(FT, TAG)                                                                  \
  PARITY_ONE(FT, TAG, Field, Explicit)                                                             \
  PARITY_ONE(FT, TAG, FieldFast, Explicit)                                                         \
  PARITY_ONE(FT, TAG, FieldWithMulCasts, Explicit)                                                 \
  PARITY_ONE(FT, TAG, SophieGermainRing, Explicit)                                                 \
  PARITY_ONE(FT, TAG, PythagoreanRing, Explicit)

PARITY_ALL_MODES(float, F32)
PARITY_ALG_MODES(float, F32)
PARITY_ALL_MODES(double, F64)
PARITY_ALG_MODES(double, F64)
#if FPSAN_HAS_FLOAT16
PARITY_ALL_MODES(_Float16, F16)
PARITY_ALG_MODES(_Float16, F16)
#endif
#if FPSAN_HAS_BF16
PARITY_ALL_MODES(__bf16, BF16)
PARITY_ALG_MODES(__bf16, BF16)
#endif
