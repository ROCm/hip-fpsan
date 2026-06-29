// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "swmmac_gfx12_w64_parts.hpp"

TEST(SwmmacGfx12W64, F32_BF8_FP8_LayoutAndFpsan) { run_fp8_case<SwBf8Fp8>(0x7000); }
