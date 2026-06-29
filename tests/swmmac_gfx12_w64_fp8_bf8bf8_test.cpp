// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "swmmac_gfx12_w64_parts.hpp"

TEST(SwmmacGfx12W64, F32_BF8_BF8_LayoutAndFpsan) { run_fp8_case<SwBf8Bf8>(0x8000); }
