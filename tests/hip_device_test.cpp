// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/hip_device_test.cpp
//
// Runs Value arithmetic on the GPU and checks that device results match
// the host bit-for-bit (the payload algebra is constexpr host/device, so they
// must agree exactly), plus that FPSan associativity holds on device. Only
// built/registered when FPSAN_ENABLE_HIP is ON.
#include "fpsan/amdgcn_matrix.hpp"
#include "fpsan/detail/subbyte_widen.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "hip_test_utils.hpp"
#include "subbyte_oracle.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

static constexpr fpsan::Conversions kCC = fpsan::Conversions::Explicit;
template <fpsan::Semantics S> using FV = fpsan::Value<float, S, fpsan::Conversions::Explicit>;

namespace {
constexpr int kHelperMaxOutputs = 512;

template <class V> __device__ std::uint32_t payload32(V v) {
  return static_cast<std::uint32_t>(v.fpsan_payload());
}

template <fpsan::Semantics S> __global__ void fpsan_helper_kernel(std::uint32_t *out) {
  int pos = 1;

  for (std::uint32_t code = 0; code < 16; ++code) {
    out[pos++] = payload32(fpsan::detail::subbyte_widen_to<float, 4, S, kCC>(code));
#if FPSAN_HAS_FLOAT16
    out[pos++] = payload32(fpsan::detail::subbyte_widen_to<_Float16, 4, S, kCC>(code));
#endif
#if FPSAN_HAS_BF16
    out[pos++] = payload32(fpsan::detail::subbyte_widen_to<__bf16, 4, S, kCC>(code));
#endif
  }
  for (std::uint32_t code = 0; code < 64; ++code) {
    out[pos++] = payload32(fpsan::detail::subbyte_widen_to<float, 6, S, kCC>(code));
#if FPSAN_HAS_FLOAT16
    out[pos++] = payload32(fpsan::detail::subbyte_widen_to<_Float16, 6, S, kCC>(code));
#endif
#if FPSAN_HAS_BF16
    out[pos++] = payload32(fpsan::detail::subbyte_widen_to<__bf16, 6, S, kCC>(code));
#endif
  }

  const float narrow_samples[] = {-13.0f, -7.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 7.0f, 13.0f};
  for (int i = 0; i < 9; ++i) {
    FV<S> v{narrow_samples[i]};
    out[pos++] = fpsan::detail::subbyte_narrow_code<4>(v);
    out[pos++] = fpsan::detail::subbyte_narrow_code<6>(v);
  }

  const float cast_samples[] = {0.0f, 1.0f, -1.0f, 1.5f, -2.25f};
#if FPSAN_HAS_FLOAT16
  for (int i = 0; i < 5; ++i) {
    using H = fpsan::Value<_Float16, S, kCC>;
    const auto f = FV<S>{cast_samples[i]};
    const auto h = H{static_cast<_Float16>(cast_samples[i])};
    out[pos++] = payload32(fpsan::cast<float>(h));
    out[pos++] = payload32(fpsan::cast<_Float16>(f));
  }
#endif
#if FPSAN_HAS_BF16
  for (int i = 0; i < 5; ++i) {
    using B = fpsan::Value<__bf16, S, kCC>;
    const auto f = FV<S>{cast_samples[i]};
    const auto b = B{static_cast<__bf16>(cast_samples[i])};
    out[pos++] = payload32(fpsan::cast<float>(b));
    out[pos++] = payload32(fpsan::cast<__bf16>(f));
  }
#endif

  const std::uint8_t e4_codes[] = {0x00u, 0x20u, 0x30u, 0x38u, 0x40u, 0xB8u};
  const std::uint8_t e5_codes[] = {0x00u, 0x30u, 0x38u, 0x3Cu, 0x40u, 0xBCu};
  for (int i = 0; i < 6; ++i) {
    using E4 = fpsan::Value<fpsan::fp8_e4m3, S, kCC>;
    const auto x = fpsan::fp8_e4m3{e4_codes[i]};
    out[pos++] = payload32(fpsan::cast<float>(E4{x}));
    out[pos++] = payload32(fpsan::cast<fpsan::fp8_e4m3>(FV<S>{static_cast<float>(x)}));
  }
  for (int i = 0; i < 6; ++i) {
    using E5 = fpsan::Value<fpsan::fp8_e5m2, S, kCC>;
    const auto x = fpsan::fp8_e5m2{e5_codes[i]};
    out[pos++] = payload32(fpsan::cast<float>(E5{x}));
    out[pos++] = payload32(fpsan::cast<fpsan::fp8_e5m2>(FV<S>{static_cast<float>(x)}));
  }

  const auto scale4 = fpsan::detail::wmma_make_scale4<0, FV<S>>(std::uint32_t{0x807F0100u});
  for (int byte = 0; byte < 4; ++byte)
    out[pos++] = payload32(scale4.select(byte));

  const std::uint64_t scale8_word = std::uint64_t{0xC0483020B8403800ull};
  const auto scale8 = fpsan::detail::wmma_make_scale8<2, FV<S>>(scale8_word);
  for (int byte = 0; byte < 8; ++byte)
    out[pos++] = payload32(scale8.select(byte));

  out[0] = static_cast<std::uint32_t>(pos);
}

template <class DstFT, int Width, fpsan::Semantics S>
void append_subbyte_widen_expected(std::vector<std::uint32_t> &expected, std::uint32_t code) {
  expected.push_back(static_cast<std::uint32_t>(
      fpsan_test::canonical_subbyte_widen_payload<DstFT, Width, S, kCC>(code)));
}

template <fpsan::Semantics S> std::vector<std::uint32_t> expected_helper_payloads() {
  std::vector<std::uint32_t> expected;
  expected.reserve(kHelperMaxOutputs);
  expected.push_back(0);

  for (std::uint32_t code = 0; code < 16; ++code) {
    append_subbyte_widen_expected<float, 4, S>(expected, code);
#if FPSAN_HAS_FLOAT16
    append_subbyte_widen_expected<_Float16, 4, S>(expected, code);
#endif
#if FPSAN_HAS_BF16
    append_subbyte_widen_expected<__bf16, 4, S>(expected, code);
#endif
  }
  for (std::uint32_t code = 0; code < 64; ++code) {
    append_subbyte_widen_expected<float, 6, S>(expected, code);
#if FPSAN_HAS_FLOAT16
    append_subbyte_widen_expected<_Float16, 6, S>(expected, code);
#endif
#if FPSAN_HAS_BF16
    append_subbyte_widen_expected<__bf16, 6, S>(expected, code);
#endif
  }

  const float narrow_samples[] = {-13.0f, -7.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 7.0f, 13.0f};
  for (float sample : narrow_samples) {
    FV<S> v{sample};
    expected.push_back(fpsan_test::canonical_subbyte_narrow_code<4>(v));
    expected.push_back(fpsan_test::canonical_subbyte_narrow_code<6>(v));
  }

  const float cast_samples[] = {0.0f, 1.0f, -1.0f, 1.5f, -2.25f};
#if FPSAN_HAS_FLOAT16
  for (float sample : cast_samples) {
    using H = fpsan::Value<_Float16, S, kCC>;
    const auto f = FV<S>{sample};
    const auto h = H{static_cast<_Float16>(sample)};
    expected.push_back(static_cast<std::uint32_t>(fpsan::cast<float>(h).fpsan_payload()));
    expected.push_back(static_cast<std::uint32_t>(fpsan::cast<_Float16>(f).fpsan_payload()));
  }
#endif
#if FPSAN_HAS_BF16
  for (float sample : cast_samples) {
    using B = fpsan::Value<__bf16, S, kCC>;
    const auto f = FV<S>{sample};
    const auto b = B{static_cast<__bf16>(sample)};
    expected.push_back(static_cast<std::uint32_t>(fpsan::cast<float>(b).fpsan_payload()));
    expected.push_back(static_cast<std::uint32_t>(fpsan::cast<__bf16>(f).fpsan_payload()));
  }
#endif

  const std::uint8_t e4_codes[] = {0x00u, 0x20u, 0x30u, 0x38u, 0x40u, 0xB8u};
  const std::uint8_t e5_codes[] = {0x00u, 0x30u, 0x38u, 0x3Cu, 0x40u, 0xBCu};
  for (std::uint8_t code : e4_codes) {
    using E4 = fpsan::Value<fpsan::fp8_e4m3, S, kCC>;
    const auto x = fpsan::fp8_e4m3{code};
    expected.push_back(static_cast<std::uint32_t>(fpsan::cast<float>(E4{x}).fpsan_payload()));
    expected.push_back(static_cast<std::uint32_t>(
        fpsan::cast<fpsan::fp8_e4m3>(FV<S>{static_cast<float>(x)}).fpsan_payload()));
  }
  for (std::uint8_t code : e5_codes) {
    using E5 = fpsan::Value<fpsan::fp8_e5m2, S, kCC>;
    const auto x = fpsan::fp8_e5m2{code};
    expected.push_back(static_cast<std::uint32_t>(fpsan::cast<float>(E5{x}).fpsan_payload()));
    expected.push_back(static_cast<std::uint32_t>(
        fpsan::cast<fpsan::fp8_e5m2>(FV<S>{static_cast<float>(x)}).fpsan_payload()));
  }

  const unsigned e8_bytes[] = {0x00u, 0x01u, 0x7Fu, 0x80u};
  for (unsigned byte : e8_bytes)
    expected.push_back(
        static_cast<std::uint32_t>(FV<S>{fpsan::detail::e8m0_to_float(byte)}.fpsan_payload()));

  const std::uint8_t scale_e4_codes[] = {0x00u, 0x38u, 0x40u, 0xB8u, 0x20u, 0x30u, 0x48u, 0xC0u};
  for (std::uint8_t code : scale_e4_codes)
    expected.push_back(static_cast<std::uint32_t>(
        FV<S>{static_cast<float>(fpsan::fp8_e4m3{code})}.fpsan_payload()));

  expected[0] = static_cast<std::uint32_t>(expected.size());
  return expected;
}

template <fpsan::Semantics S> void run_helper_device_check() {
  std::uint32_t *dout;
  HIP_CHECK(hipMalloc(&dout, kHelperMaxOutputs * sizeof(std::uint32_t)));
  HIP_CHECK(hipMemset(dout, 0, kHelperMaxOutputs * sizeof(std::uint32_t)));
  fpsan_helper_kernel<S><<<1, 1>>>(dout);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<std::uint32_t> got(kHelperMaxOutputs);
  HIP_CHECK(hipMemcpy(got.data(), dout, kHelperMaxOutputs * sizeof(std::uint32_t),
                      hipMemcpyDeviceToHost));
  (void)hipFree(dout);

  const auto expected = expected_helper_payloads<S>();
  ASSERT_EQ(got[0], expected.size()) << "S=" << int(S);
  for (std::size_t i = 1; i < expected.size(); ++i)
    EXPECT_EQ(got[i], expected[i]) << "S=" << int(S) << " helper payload index " << i;
}
} // namespace

template <fpsan::Semantics S>
__global__ void fpsan_kernel(const float *a, const float *b, const float *c,
                             std::uint32_t *embed_payload, std::uint32_t *assoc_ok,
                             std::uint32_t *exp_hom_ok, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  FV<S> A(a[i]), B(b[i]), C(c[i]);
  embed_payload[i] = A.fpsan_payload();
  assoc_ok[i] = ((A + B) + C == A + (B + C)) ? 1u : 0u;
  exp_hom_ok[i] = (fpsan::exp(A + B) == fpsan::exp(A) * fpsan::exp(B)) ? 1u : 0u;
}

TEST(HipDevice, MatchesHostAndPreservesIdentities) {
  int ndev = 0;
  hipError_t e = hipGetDeviceCount(&ndev);
  if (e != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device available";

  // Deterministic pseudo-random inputs (see test_random.hpp). The
  // device-vs-host and identity checks hold for any finite values; quarters
  // give broad, exactly-representable coverage.
  const int n = 64;
  std::vector<float> a(n), b(n), c(n);
  std::mt19937 rng = fpsan_test::make_rng();
  for (int i = 0; i < n; ++i) {
    a[i] = fpsan_test::pick_quarter<float>(rng, -36, 36); // -9 .. 9
    b[i] = fpsan_test::pick_quarter<float>(rng, -36, 36);
    c[i] = fpsan_test::pick_quarter<float>(rng, -36, 36);
  }

  float *da, *db, *dc;
  std::uint32_t *dembed, *dassoc, *dexp;
  HIP_CHECK(hipMalloc(&da, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&db, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&dc, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&dembed, n * sizeof(std::uint32_t)));
  HIP_CHECK(hipMalloc(&dassoc, n * sizeof(std::uint32_t)));
  HIP_CHECK(hipMalloc(&dexp, n * sizeof(std::uint32_t)));
  HIP_CHECK(hipMemcpy(da, a.data(), n * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(db, b.data(), n * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dc, c.data(), n * sizeof(float), hipMemcpyHostToDevice));

  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr fpsan::Semantics S = decltype(sem)::value;
    fpsan_kernel<S><<<dim3((n + 63) / 64), dim3(64)>>>(da, db, dc, dembed, dassoc, dexp, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<std::uint32_t> embed(n), assoc(n), exph(n);
    HIP_CHECK(hipMemcpy(embed.data(), dembed, n * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(assoc.data(), dassoc, n * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(exph.data(), dexp, n * sizeof(std::uint32_t), hipMemcpyDeviceToHost));

    // exp is a genuine homomorphism in Triton and the two-moduli algebraic
    // models, but a tag in the Field variants (no exp channel) -- so
    // only assert it where it is one. embed-matches-host and associativity hold
    // for every semantics.
    constexpr bool exp_is_hom =
        (S == fpsan::Semantics::Triton) ||
        (fpsan::detail::is_algebraic_semantics(S) && FV<S>::alg_cfg().two_moduli);
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(embed[i], FV<S>(a[i]).fpsan_payload()) << "device embed != host @" << i;
      EXPECT_EQ(assoc[i], 1u) << "associativity failed on device @" << i;
      if constexpr (exp_is_hom)
        EXPECT_EQ(exph[i], 1u) << "exp homomorphism failed on device @" << i;
    }
  });

  (void)hipFree(da);
  (void)hipFree(db);
  (void)hipFree(dc);
  (void)hipFree(dembed);
  (void)hipFree(dassoc);
  (void)hipFree(dexp);
}

TEST(HipDevice, SubbyteCastAndScaleHelpersMatchHost) {
  int ndev = 0;
  hipError_t e = hipGetDeviceCount(&ndev);
  if (e != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device available";

  fpsan_test::for_each_fpsan_semantics_all_variants(
      [&](auto sem) { run_helper_device_check<decltype(sem)::value>(); });
}
