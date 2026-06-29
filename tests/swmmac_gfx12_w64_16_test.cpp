// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "swmmac_gfx12_w64_parts.hpp"

TEST(SwmmacGfx12W64, F16_F16_LayoutAndFpsan) { run_h_case<SwF16F16>(0x3000); }
TEST(SwmmacGfx12W64, BF16_BF16_LayoutAndFpsan) { run_h_case<SwBF16BF16>(0x4000); }
