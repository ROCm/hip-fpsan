#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Regression test: the fpsan device path must build & LINK under -nogpulib
# (no ROCm device library), so it can target environments without the bitcode
# runtime (e.g. the rocjitsu simulator).
#
# Background: fpsan's own device code contains no assert()/printf, but
# <hip/hip_runtime.h> defines a *weak* device __assert_fail that references the
# ockl printf helpers (__ockl_fprintf_*). Under -nogpulib those helpers are
# absent, so the amdgcn link fails UNLESS the unreferenced __assert_fail is
# dead-stripped. -O2 strips it via IR DCE, but a robust -nogpulib build (any
# optimization level, incl. -O0 debug) needs per-function sections plus
# --gc-sections so the linker can drop it. This test pins that recipe.
#
# Honors: FPSAN_HIP_CLANG, FPSAN_ROCM_PATH, FPSAN_GPU_ARCH. Skips (exit 0) if no
# HIP toolchain is found, so it is safe in a pure-C++ CI.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
inc="$here/../include"
src="$here/algebraic_device_test.cpp"
arch="${FPSAN_GPU_ARCH:-gfx1201}"

# Locate a ROCm clang (the amdclang++ wrapper in some SDKs is a broken python
# shim, so prefer the real clang++ binary under the SDK).
clang="${FPSAN_HIP_CLANG:-}"
if [[ -z "$clang" ]]; then
  for c in \
    "$HOME"/therock_venv/lib/python*/site-packages/_rocm_sdk_core/lib/llvm/bin/clang++ \
    "$HOME"/therock_venv/lib/llvm/bin/clang++; do
    [[ -x "$c" ]] && clang="$c" && break
  done
fi
if [[ -z "$clang" || ! -x "$clang" ]]; then
  echo "SKIP: no ROCm clang found (set FPSAN_HIP_CLANG to run this test)."
  exit 0
fi

# ROCm root: explicit, else two levels up from .../lib/llvm/bin/clang++.
rocm="${FPSAN_ROCM_PATH:-}"
if [[ -z "$rocm" ]]; then
  rocm="$(cd "$(dirname "$clang")/../../.." && pwd)"
fi

out="$(mktemp -d)/algd.o"
echo "fpsan -nogpulib build test: $clang  arch=$arch"
"$clang" -x hip --offload-arch="$arch" --rocm-path="$rocm" \
  -nogpulib -O0 -ffunction-sections -std=c++17 -I "$inc" \
  -Xoffload-linker --gc-sections \
  -c "$src" -o "$out"
rc=$?
if [[ $rc -eq 0 && -s "$out" ]]; then
  echo "PASS: device code links under -nogpulib ($(stat -c%s "$out") bytes)."
  exit 0
fi
echo "FAIL: -nogpulib device build broken (rc=$rc)."
exit 1
