// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "swmmac_gfx12_w64_parts.hpp"

TEST(SwmmacGfx12W64, F32_F16_LayoutAndFpsan) { run_h_case<SwF32F16>(0x1000); }
TEST(SwmmacGfx12W64, F32_BF16_LayoutAndFpsan) { run_h_case<SwF32BF16>(0x2000); }
