// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// Host-side test of csrc/pth_attn/tile_traits.cuh (built and run by
// tests/cpu/test_tile_traits.py with g++; no CUDA needed).
//
// Two halves:
//  1. static_asserts that pin the tile table for every row it lists (a
//     wrong formula fails to COMPILE);
//  2. one "key=value" line per (D, WM) for all nine shipped configurations,
//     which the Python runner recomputes with independent arithmetic.
//
// The negative controls (geometries that must not compile) are separate
// snippets in the runner, so that this file always builds.

#include <cstdint>
#include <cstdio>

#include "tile_traits.cuh"

namespace {

using pth_attn::Traits;
using Io16 = uint16_t;  // stands for bf16/fp16: only its size matters here

// Tile table, row by row: Q smem, stage, total, CTAs/SM, compute sub-tile.
template <int D, int WM, int Q, int STAGE, int TOTAL, int CTAS, int SUB>
constexpr bool table_row() {
  using T = Traits<D, WM, Io16>;
  static_assert(T::kQBytes == Q, "Q smem differs from the tile table");
  static_assert(T::kStageBytes == STAGE,
                "stage bytes differ from the tile table");
  static_assert(T::kStagesN * T::kStageBytes == 2 * STAGE, "2 stages");
  static_assert(T::kSmemBytes == TOTAL,
                "total smem differs from the tile table");
  static_assert(T::kCtasPerSm == CTAS, "CTAs/SM differ from the tile table");
  static_assert(T::kSub == SUB, "compute sub-tile differs from the tile table");
  return true;
}
static_assert(table_row<256, 4, 32768, 33280, 99328, 1, 32>());
static_assert(table_row<256, 2, 16384, 33280, 82944, 1, 32>());
static_assert(table_row<256, 1, 8192, 33280, 74752, 1, 16>());
static_assert(table_row<128, 4, 16384, 16896, 50176, 2, 64>());
static_assert(table_row<64, 4, 8192, 8704, 25600, 3, 64>());

// Two CTAs of D = 128 fill the 102,400-byte shared-memory budget exactly.
static_assert(2 * (Traits<128, 4, Io16>::kSmemBytes +
                   pth_attn::kSmemReservedPerBlock) ==
              pth_attn::kSmemPerSm);
// O per warp is 16 * D * 4 B / 32 lanes = D/2 registers (128 at
// D = 256).
static_assert(Traits<256, 4, Io16>::kORegs == 128);
// The decode in-CTA merge at WN = 4, D = 256 is 3 x (16 x 256 x 4 B)
// = 48 KB of O (plus 2 floats per row), staged in the 66560-B stage area.
static_assert(Traits<256, 1, Io16>::kMergeBytes ==
              3 * 16 * 256 * 4 + 3 * 16 * 8);
// Register budgets that keep the CTAs resident (D = 64 runs 3 CTAs
// per SM, 168 x 128 x 3 = 64,512 <= 65,536, which is tight).
static_assert(Traits<64, 4, Io16>::kRegBudget == 168);
static_assert(Traits<128, 4, Io16>::kRegBudget == 255);
static_assert(Traits<256, 4, Io16>::kRegBudget == 255);

template <int D, int WM>
void print_row() {
  using T = Traits<D, WM, Io16>;
  std::printf(
      "D=%d WM=%d WN=%d rows=%d N=%d stages=%d tokens_per_warp=%d sub=%d "
      "sub_steps=%d q_bytes=%d stage_bytes=%d smem_bytes=%d merge_bytes=%d "
      "ctas_per_sm=%d reg_budget=%d o_regs=%d s_regs=%d out_bytes=%d\n",
      D, WM, T::kWn, T::kRows, T::kN, T::kStagesN, T::kTokensPerWarp, T::kSub,
      T::kSubSteps, T::kQBytes, T::kStageBytes, T::kSmemBytes, T::kMergeBytes,
      T::kCtasPerSm, T::kRegBudget, T::kORegs, T::kSRegs, T::kOutElemBytes);
}

}  // namespace

int main() {
  print_row<256, 4>();
  print_row<256, 2>();
  print_row<256, 1>();
  print_row<128, 4>();
  print_row<128, 2>();
  print_row<128, 1>();
  print_row<64, 4>();
  print_row<64, 2>();
  print_row<64, 1>();
  return 0;
}
