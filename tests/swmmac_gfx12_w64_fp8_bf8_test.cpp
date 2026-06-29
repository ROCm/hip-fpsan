// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "swmmac_gfx12_w64_parts.hpp"

TEST(SwmmacGfx12W64, F32_FP8_BF8_LayoutAndFpsan) { run_fp8_case<SwFp8Bf8>(0x6000); }
