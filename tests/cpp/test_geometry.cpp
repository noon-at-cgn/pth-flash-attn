// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// Teaching tests for csrc/pth_attn/cache_geometry.cuh, on the host (g++).
//
// Build and run: tests/cpp/run_geometry.sh (g++ build and run; also compiles
// this file with nvcc for sm_80/86/89, where the __global__ probe at the end
// instantiates every function in device code).
//
// The same file built with -DPTH_GEOMETRY_SHARED -shared is the ctypes library
// tests/cpu/test_geometry_vs_reference.py loads to compare the header with
// pth_testing.reference (real vLLM-layout caches) and the planner. Why
// ctypes rather than JSON golden files or a CLI: the Python side passes the
// real torch data pointers and view strides, so the C++ code reads the actual
// bytes of a cache built with vLLM's layout, including views whose byte
// offsets exceed 2^31 (lazily committed memory), with nothing serialised in
// between that could itself be wrong.
//
// Each test states the property it protects and why; the safety-critical ones
// carry a negative control that must fail.

#include "cache_geometry.cuh"
#include "mask.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace pth_attn;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    if (g_failures <= 20) std::printf("  FAIL: %s\n", what.c_str());
  }
}

// ======================================================== 1. FastDivmod
// Property: div/divmod equal integer division for every numerator the kernel
// can pass (n < 2^31) and every divisor 1..2^31. Why: a wrong (block, slot)
// silently reads another token's K/V.
//
// Strategy: the proof in the header rests on three premises per divisor
// (m < 2^32, 0 <= e < d, e*(2^31-1) < 2^p). They are checked for EVERY d in
// [1, 2^16] (the supported range: block sizes 1..2^16) and for d near 2^31.
// Exhaustive n < 2^20 (the supported kv <= 2^20) for d in 1..1056 (every block
// size up to 1056 = the largest supported), all powers of two up to 2^16,
// and 65535/65536. The top of the domain (n = k*d - 1, k*d around 2^31) for
// every d in [1, 2^16]. Negative control: one bit less of precision
// (p = 30 + l) must produce a wrong quotient somewhere in the domain.
void test_fastdivmod() {
  std::printf("test_fastdivmod\n");
  int bad = 0;
  auto premises = [&](uint32_t d) {
    const FastDivmod f = FastDivmod::make(d);
    if (!f.valid() || f.divisor != d) return false;
    if (d == 1) return true;
    const uint32_t p = f.shift + 32;
    const unsigned __int128 md =
        static_cast<unsigned __int128>(f.multiplier) * d;
    const unsigned __int128 two_p = static_cast<unsigned __int128>(1) << p;
    if (md < two_p) return false;  // m >= 2^p / d
    const unsigned __int128 e = md - two_p;
    if (e >= d) return false;                    // m is the smallest such
    return e * kFastDivmodMaxNumerator < two_p;  // the proof's key inequality
  };
  for (uint32_t d = 1; d <= (1u << 16); ++d) bad += !premises(d);
  for (uint32_t d = (1u << 31) - 70; d <= (1u << 31); ++d) bad += !premises(d);
  check(bad == 0, "premises for every d in [1, 2^16] and near 2^31");

  std::vector<uint32_t> exhaustive;
  for (uint32_t d = 1; d <= 1056; ++d) exhaustive.push_back(d);
  for (uint32_t k = 11; k <= 16; ++k) exhaustive.push_back(1u << k);
  exhaustive.push_back(65535);
  bad = 0;
  for (uint32_t d : exhaustive) {
    const FastDivmod f = FastDivmod::make(d);
    for (uint32_t n = 0; n < (1u << 20); ++n) {
      uint32_t q, r;
      f.divmod(n, q, r);
      if (q != n / d || r != n % d) ++bad;
    }
  }
  check(bad == 0, "exhaustive n < 2^20 for d in 1..1056 and 2^11..2^16");

  bad = 0;
  for (uint32_t d = 1; d <= (1u << 16); ++d) {
    const FastDivmod f = FastDivmod::make(d);
    const uint32_t top = kFastDivmodMaxNumerator;
    const uint32_t kmax = top / d;
    for (uint32_t k = kmax > 2 ? kmax - 2 : 0; k <= kmax; ++k) {
      const uint64_t base = static_cast<uint64_t>(k) * d;
      for (uint64_t n : {base, base + d - 1, base == 0 ? 0 : base - 1}) {
        if (n > top) continue;
        const uint32_t un = static_cast<uint32_t>(n);
        if (f.div(un) != un / d) ++bad;
      }
    }
    if (f.div(top) != top / d) ++bad;
  }
  check(bad == 0, "top of the domain for every d in [1, 2^16]");

  std::mt19937_64 rng(1);
  bad = 0;
  for (int i = 0; i < 2000000; ++i) {
    const uint32_t d = 1 + static_cast<uint32_t>(rng() % kFastDivmodMaxDivisor);
    const uint32_t n = static_cast<uint32_t>(rng() & kFastDivmodMaxNumerator);
    if (FastDivmod::make(d).div(n) != n / d) ++bad;
  }
  check(bad == 0, "2e6 random (n, d) over the whole domain");
  check(!FastDivmod::make(0).valid() &&
            !FastDivmod::make(kFastDivmodMaxDivisor + 1).valid(),
        "d = 0 and d > 2^31 are rejected");

  // Negative control: p = 30 + l (one bit less) must fail for some (n, d).
  bool found = false;
  for (uint32_t d = 3; d <= 2000 && !found; ++d) {
    uint32_t l = 0;
    while ((uint64_t{1} << l) < d) ++l;
    const uint32_t p = 30 + l;
    if (p < 32) continue;
    const uint32_t m = static_cast<uint32_t>(((uint64_t{1} << p) + d - 1) / d);
    for (uint32_t k = kFastDivmodMaxNumerator / d; k > 0 && !found;
         k -= (k > 4096 ? 4096 : k)) {
      const uint32_t n = k * d - 1;  // largest remainder: the worst case
      if ((umulhi32(n, m) >> (p - 32)) != n / d) found = true;
    }
  }
  check(found, "negative control: one bit less of precision is wrong");
}

// ======================================================== 2. stream widths
// Property: the width is the largest of {16, 8, 4} dividing the base, the row
// and every stride that is multiplied by a non-zero index, and size-1
// dimensions are ignored. Why: a too-wide cp.async faults or reads a wrong
// line on misaligned addresses; a too-narrow one costs bandwidth.
// Today's vLLM cell [K D | ks 4 | V D | vs 4] gives K 8, V 4, scales 4 for
// every D; #56164's decoupled caches give K/V 16, scales 4.
void inline_views(int32_t D, int32_t BS, int32_t Hkv, bool nhd, int64_t pad,
                  int64_t base_off, StreamGeometry st[kNumStreams],
                  int64_t sizes[3]) {
  const int64_t C = 2 * (D + 4);
  const int64_t page = static_cast<int64_t>(BS) * Hkv * C + pad;
  const int64_t s_tok = nhd ? Hkv * C : C;
  const int64_t s_head = nhd ? C : static_cast<int64_t>(BS) * C;
  sizes[0] = 1000;
  sizes[1] = BS;
  sizes[2] = Hkv;
  // A fake base address with the given offset from a 4096-aligned allocation.
  const uintptr_t base = 4096 * 16 + base_off;
  const int64_t byte_strides[3] = {page, s_tok, s_head};
  const int64_t offs[kNumStreams] = {0, D + 4, D, 2 * D + 4};
  const int32_t rows[kNumStreams] = {D, D, 4, 4};
  for (int s = 0; s < kNumStreams; ++s) {
    const int64_t elem = s < 2 ? 1 : 4;
    const int64_t es[3] = {byte_strides[0] / elem, byte_strides[1] / elem,
                           byte_strides[2] / elem};
    st[s] = make_stream(reinterpret_cast<const void*>(base + offs[s]), sizes,
                        es, elem, rows[s]);
  }
}

void test_stream_widths() {
  std::printf("test_stream_widths\n");
  for (int32_t D : {64, 128, 256}) {
    for (bool nhd : {true, false}) {
      for (int32_t BS : {1, 16, 64, 528, 816, 1056}) {
        StreamGeometry st[kNumStreams];
        int64_t sizes[3];
        inline_views(D, BS, 2, nhd, 0, 0, st, sizes);
        const std::string tag = "D=" + std::to_string(D) +
                                (nhd ? " NHD" : " HND") +
                                " BS=" + std::to_string(BS);
        check(st[kStreamK].width == 8, tag + ": K width 8");
        check(st[kStreamV].width == 4, tag + ": V width 4");
        check(st[kStreamKScale].width == 4 && st[kStreamVScale].width == 4,
              tag + ": scale width 4");
        // Base only 4-aligned (storage offset 4): K drops to 4.
        inline_views(D, BS, 2, nhd, 0, 4, st, sizes);
        check(st[kStreamK].width == 4, tag + ": K width 4 at base+4");
      }
    }
  }
  // Decoupled (#56164-style): K/V [NB, BS, Hkv, D] contiguous, scales
  // [NB, BS, Hkv] f32 contiguous: K/V 16, scales 4.
  for (int32_t D : {64, 128, 256}) {
    const int64_t sizes[3] = {100, 816, 2};
    const int64_t kv_strides[3] = {816LL * 2 * D, 2LL * D, D};
    const int64_t sc_strides[3] = {816LL * 2, 2, 1};
    const StreamGeometry k =
        make_stream(reinterpret_cast<const void*>(uintptr_t{1} << 20), sizes,
                    kv_strides, 1, D);
    const StreamGeometry s =
        make_stream(reinterpret_cast<const void*>(uintptr_t{1} << 21), sizes,
                    sc_strides, 4, 4);
    check(k.width == 16 && s.width == 4,
          "decoupled D=" + std::to_string(D) + ": K/V 16, scales 4");
  }
  // A size-1 head dimension with an odd stride does not narrow the width,
  // but the same odd stride on a dimension of size 2 does (negative control).
  {
    const int64_t sz1[3] = {10, 16, 1}, sz2[3] = {10, 16, 2};
    const int64_t bs[3] = {16 * 528, 528, 3};
    const uint64_t base = 1u << 20;
    check(stream_width(base, sz1, bs, 3, 256) == 16,
          "size-1 dimension stride ignored");
    check(stream_width(base, sz2, bs, 3, 256) == 0,
          "negative control: stride 3 on a real dimension gives no width");
  }
  check(stream_width(1u << 20, nullptr, nullptr, 0, 2) == 0,
        "row bytes 2 has no cp.async width");
}

// ================================================ 3. addresses: stride fuzz
// Property: the byte address of (page, slot, head) comes only from the view's
// strides, so it is right for ANY layout: random physical orders of
// (block, slot, head), random padding after every level, random base offset.
// Why: vLLM #49716 (offsets derived from shapes broke when the layout
// changed). The fuzz fills each layout by walking it with a cursor (no stride
// arithmetic), so the expected bytes are independent of the formula under
// test. Negative control: an implementation that derives strides from the
// shapes (contiguous NHD) must fail on the same fuzz set.
struct FuzzLayout {
  int64_t sizes[3];    // NB, BS, Hkv
  int64_t strides[3];  // bytes, as the generator laid them out
  int32_t row;         // row bytes
  int64_t base_off;
  std::vector<uint8_t> mem;
};

uint8_t pattern(int64_t b, int64_t t, int64_t h, int64_t k) {
  uint64_t x = static_cast<uint64_t>(b) * 0x9E3779B97F4A7C15ull ^
               static_cast<uint64_t>(t) * 0xC2B2AE3D27D4EB4Full ^
               static_cast<uint64_t>(h) * 0x165667B19E3779F9ull ^
               static_cast<uint64_t>(k) * 0x27D4EB2F165667C5ull;
  x ^= x >> 29;
  return static_cast<uint8_t>(x * 0xBF58476D1CE4E5B9ull >> 56);
}

FuzzLayout make_fuzz(std::mt19937_64& rng, int64_t NB, int64_t BS, int64_t H,
                     int32_t row) {
  FuzzLayout L;
  L.sizes[0] = NB;
  L.sizes[1] = BS;
  L.sizes[2] = H;
  L.row = row;
  L.base_off = 4 * static_cast<int64_t>(rng() % 8);
  int order[3] = {0, 1, 2};  // outermost first
  std::shuffle(order, order + 3, rng);
  int64_t pad[3];
  for (int64_t& p : pad) p = 4 * static_cast<int64_t>(rng() % 5);
  // Extent of one element of each level, innermost (the row) outward.
  int64_t inner = row + 4 * static_cast<int64_t>(rng() % 3);
  int64_t level_stride[3];
  for (int lvl = 2; lvl >= 0; --lvl) {
    const int dim = order[lvl];
    level_stride[lvl] = inner;
    inner = inner * L.sizes[dim] + pad[lvl];
  }
  for (int lvl = 0; lvl < 3; ++lvl) L.strides[order[lvl]] = level_stride[lvl];
  L.mem.assign(static_cast<size_t>(L.base_off + inner), 0xEE);
  // Walk the layout with a cursor: the row of (b, t, h) is written wherever
  // the walk is when it reaches that index. No multiplication by a stride.
  int64_t cursor = L.base_off;
  int64_t idx[3];
  for (idx[order[0]] = 0; idx[order[0]] < L.sizes[order[0]]; ++idx[order[0]]) {
    const int64_t c0 = cursor;
    for (idx[order[1]] = 0; idx[order[1]] < L.sizes[order[1]];
         ++idx[order[1]]) {
      const int64_t c1 = cursor;
      for (idx[order[2]] = 0; idx[order[2]] < L.sizes[order[2]];
           ++idx[order[2]]) {
        for (int32_t k = 0; k < row; ++k) {
          L.mem[static_cast<size_t>(cursor + k)] =
              pattern(idx[0], idx[1], idx[2], k);
        }
        cursor += level_stride[2];
      }
      cursor = c1 + level_stride[1];
    }
    cursor = c0 + level_stride[0];
  }
  return L;
}

void test_stride_fuzz() {
  std::printf("test_stride_fuzz\n");
  std::mt19937_64 rng(49716);
  int64_t layouts = 0, geom_bad = 0, shape_bad_layouts = 0;
  for (int it = 0; it < 300; ++it) {
    const int64_t NB = 1 + static_cast<int64_t>(rng() % 5);
    const int64_t BS = 1 + static_cast<int64_t>(rng() % 20);
    const int64_t H = 1 + static_cast<int64_t>(rng() % 4);
    const int32_t row = 4 * (1 + static_cast<int32_t>(rng() % 8));
    FuzzLayout L = make_fuzz(rng, NB, BS, H, row);
    const StreamGeometry s =
        make_stream(L.mem.data() + L.base_off, L.sizes, L.strides, 1, row);
    bool shape_wrong = false;
    for (int64_t b = 0; b < NB; ++b) {
      for (int64_t t = 0; t < BS; ++t) {
        for (int32_t h = 0; h < H; ++h) {
          const char* a = s.address(b, static_cast<uint32_t>(t), h);
          const char* lo = reinterpret_cast<const char*>(L.mem.data());
          if (a < lo || a + row > lo + L.mem.size()) {  // wrong, not a crash
            geom_bad += row;
            shape_wrong = true;
            continue;
          }
          for (int32_t k = 0; k < row; ++k) {
            if (static_cast<uint8_t>(a[k]) != pattern(b, t, h, k)) ++geom_bad;
          }
          // Negative control: strides derived from the shapes (contiguous
          // [NB, BS, H, row], the #49716 bug class).
          const int64_t shape_off = ((b * BS + t) * H + h) * row;
          const uint8_t* sa = L.mem.data() + L.base_off + shape_off;
          if (sa + row > L.mem.data() + L.mem.size() ||
              std::memcmp(sa, a, static_cast<size_t>(row)) != 0) {
            shape_wrong = true;
          }
        }
      }
    }
    ++layouts;
    shape_bad_layouts += shape_wrong;
  }
  check(geom_bad == 0, "geometry reads the right bytes on all fuzz layouts");
  check(shape_bad_layouts > layouts / 2,
        "negative control: shape-derived offsets are wrong on " +
            std::to_string(shape_bad_layouts) + "/" + std::to_string(layouts) +
            " layouts (must be most)");
}

// Property: offsets are exact int64 for pages whose byte offset exceeds 2^31
// (production pools: page 2,531 at 848,640 B/page already passes 2^31).
// Checked against 128-bit arithmetic; negative control: the same sum in
// 32-bit arithmetic is wrong.
void test_int64_offsets() {
  std::printf("test_int64_offsets\n");
  StreamGeometry st[kNumStreams];
  int64_t sizes[3];
  inline_views(256, 816, 2, true, 64, 0, st, sizes);
  int bad = 0;
  bool int32_wrong = false;
  for (int64_t page : {int64_t{0}, int64_t{2530}, int64_t{2531},
                       int64_t{1} << 20, (int64_t{1} << 31) - 1}) {
    for (uint32_t slot : {0u, 1u, 815u}) {
      for (int32_t h : {0, 1}) {
        const StreamGeometry& s = st[kStreamV];
        const __int128 want = static_cast<__int128>(page) * s.s_block +
                              static_cast<__int128>(slot) * s.s_token +
                              static_cast<__int128>(h) * s.s_head;
        if (static_cast<__int128>(s.offset(page, slot, h)) != want) ++bad;
        const uint32_t narrow =
            static_cast<uint32_t>(page) * static_cast<uint32_t>(s.s_block) +
            slot * static_cast<uint32_t>(s.s_token) +
            static_cast<uint32_t>(h) * static_cast<uint32_t>(s.s_head);
        if (static_cast<__int128>(static_cast<int32_t>(narrow)) != want) {
          int32_wrong = true;
        }
      }
    }
  }
  check(bad == 0, "int64 offsets equal 128-bit reference up to page 2^31-1");
  check(int32_wrong, "negative control: 32-bit offsets overflow");
}

// Property: token_page reads block_table[req][pos / BS] with the view's row
// stride and returns slot pos % BS, for any BS (including 1 and non powers of
// two such as 528/816/1056).
void test_token_page() {
  std::printf("test_token_page\n");
  std::mt19937_64 rng(7);
  int bad = 0;
  for (int32_t BS : {1, 16, 64, 528, 816, 1056}) {
    const int64_t row_stride = 37;  // padded block_table rows
    std::vector<int32_t> table(4 * row_stride);
    for (auto& v : table) v = static_cast<int32_t>(rng() % 100000);
    KernelParams p{};
    p.block_table = table.data();
    p.bt_row_stride = row_stride;
    p.block_size = FastDivmod::make(static_cast<uint32_t>(BS));
    for (int32_t req = 0; req < 4; ++req) {
      for (uint32_t pos = 0; pos < 37u * static_cast<uint32_t>(BS);
           pos += 1 + static_cast<uint32_t>(rng() % 7)) {
        // Check the block index before token_page uses it, so a wrong
        // division fails here instead of reading outside the table.
        if (p.block_size.div(pos) != pos / static_cast<uint32_t>(BS)) {
          ++bad;
          continue;
        }
        uint32_t slot = 0;
        const int64_t page = token_page(p, req, pos, slot);
        if (page != table[req * row_stride + pos / BS] || slot != pos % BS) {
          ++bad;
        }
      }
    }
  }
  check(bad == 0, "token_page = (block_table[req][pos/BS], pos%BS)");
}

// ============================================== 4. splits, rows and causal
// Property: split ranges are disjoint, cover [0, T), differ by <= 1 tile, and
// equal [floor(sT/S), floor((s+1)T/S)) (planner.split_tile_range; compared
// directly with the planner in the Python test). int64 products: T up to
// 2^26, S up to 1024.
void test_split_partition() {
  std::printf("test_split_partition\n");
  int bad = 0;
  std::mt19937_64 rng(3);
  auto one = [&](int32_t T, int32_t S) {
    int32_t prev_end = 0, mn = INT32_MAX, mx = 0;
    for (int32_t s = 0; s < S; ++s) {
      int32_t b, e;
      split_tile_range(s, S, T, b, e);
      if (b != prev_end || e < b) ++bad;
      if (b != static_cast<int32_t>(static_cast<int64_t>(s) * T / S)) ++bad;
      prev_end = e;
      mn = std::min(mn, e - b);
      mx = std::max(mx, e - b);
    }
    if (prev_end != T || mx - mn > 1) ++bad;
  };
  for (int32_t T = 0; T <= 300; ++T) {
    for (int32_t S = 1; S <= 70; ++S) one(T, S);
  }
  for (int i = 0; i < 200; ++i) {
    one(static_cast<int32_t>(rng() % (1u << 26)),
        1 + static_cast<int32_t>(rng() % 1024));
  }
  check(bad == 0, "split partition: cover, disjoint, balanced, formula");
}

// Property: BQ = floor(16*wm/G) is the most query tokens whose G heads fit in
// 16*wm rows, and row r maps to (token r / G, head r % G) bijectively on the
// first num_tok*G rows; later rows are padding. Why: a wrong mapping writes
// head g's output into head g' (caught by the bound in the GPU correctness
// grid, but cheaper here).
void test_gqa_packing() {
  std::printf("test_gqa_packing\n");
  int bad = 0;
  for (int32_t wm : {1, 2, 4}) {
    for (int32_t G = 1; G <= kGeomRowsPerWarp * wm; ++G) {
      const int32_t bq = bq_capacity(wm, G);
      const int32_t rows = kGeomRowsPerWarp * wm;
      if (bq < 1 || bq * G > rows || (bq + 1) * G <= rows) ++bad;
      const FastDivmod g = FastDivmod::make(static_cast<uint32_t>(G));
      for (int32_t num_tok = 1; num_tok <= bq; ++num_tok) {
        std::vector<int> seen(static_cast<size_t>(num_tok * G), 0);
        for (int32_t r = 0; r < rows; ++r) {
          int32_t t, h;
          const bool valid =
              row_to_token_head(static_cast<uint32_t>(r), g, num_tok, t, h);
          if (valid != (r < num_tok * G)) ++bad;
          if (valid) {
            if (t * G + h != r || h >= G) ++bad;
            if (t >= 0 && t < num_tok && h >= 0 && h < G) {
              ++seen[static_cast<size_t>(t * G + h)];
            }
          }
        }
        for (int c : seen) bad += (c != 1);
      }
    }
  }
  check(bad == 0, "BQ capacity and token-major row mapping");
}

// Property: q-tile x maps to the request and token range a brute-force walk
// over the planner's tile numbering gives; every query token of every request
// is in exactly one non-empty tile with x < T // bq + num_reqs (the planner's
// grid.x, which uses a bq that is <= this one). Includes empty requests.
void test_row_tiles() {
  std::printf("test_row_tiles\n");
  std::mt19937_64 rng(11);
  int bad = 0;
  for (int it = 0; it < 3000; ++it) {
    const int32_t R = 1 + static_cast<int32_t>(rng() % 9);
    std::vector<int32_t> qsl(static_cast<size_t>(R + 1), 0);
    std::vector<int32_t> lens(static_cast<size_t>(R));
    for (int32_t r = 0; r < R; ++r) {
      const int32_t q =
          (rng() % 5 == 0) ? 0 : 1 + static_cast<int32_t>(rng() % 40);
      qsl[static_cast<size_t>(r + 1)] = qsl[static_cast<size_t>(r)] + q;
      lens[static_cast<size_t>(r)] = q + static_cast<int32_t>(rng() % 300);
    }
    const int32_t T = qsl[static_cast<size_t>(R)];
    const int32_t bq = 1 + static_cast<int32_t>(rng() % 16);
    const FastDivmod bd = FastDivmod::make(static_cast<uint32_t>(bq));
    const int32_t grid_x = T / bq + R;
    std::vector<int> covered(static_cast<size_t>(T), 0);
    for (int32_t x = 0; x < grid_x + 3; ++x) {
      RowTile t{};
      if (!row_tile(qsl.data(), lens.data(), bd, bq, R, x, t)) continue;
      if (x >= grid_x) ++bad;  // a tile with work beyond the grid
      // Brute force: walk requests, numbering tiles as qsl[r]/bq + r + j.
      int32_t want_r = -1, want_j = -1;
      for (int32_t r = 0; r < R; ++r) {
        const int32_t first = qsl[static_cast<size_t>(r)] / bq + r;
        const int32_t next = qsl[static_cast<size_t>(r + 1)] / bq + r + 1;
        if (x >= first && x < next) {
          want_r = r;
          want_j = x - first;
        }
      }
      if (t.req != want_r || t.tok_begin != want_j * bq) ++bad;
      if (t.num_tok < 1 || t.num_tok > bq ||
          t.tok_begin + t.num_tok > t.q_len) {
        ++bad;
      }
      for (int32_t i = 0; i < t.num_tok; ++i) {
        ++covered[static_cast<size_t>(t.q_start + t.tok_begin + i)];
      }
    }
    for (int c : covered) bad += (c != 1);
  }
  check(bad == 0, "row tiles cover every query token once, inside the grid");
}

// Property: kv_limit is bottom-right causal (token i of q_len over seq_len
// sees j <= seq_len - q_len + i); the CTA range end and min limit come from
// its last and first token; tile_needs_mask is EXACT: true iff some
// (row, kv token) pair of the tile is masked. Why: an unmasked boundary tile
// leaks future tokens; a needlessly masked tile costs instructions in every
// tile. Negative control: top-left alignment (j <= i) disagrees.
void test_causal() {
  std::printf("test_causal\n");
  std::mt19937_64 rng(5);
  int bad = 0;
  bool topleft_differs = false;
  for (int it = 0; it < 2000; ++it) {
    const int32_t q_len = 1 + static_cast<int32_t>(rng() % 50);
    const int32_t seq_len = q_len + static_cast<int32_t>(rng() % 400);
    const int32_t bq = 1 + static_cast<int32_t>(rng() % 16);
    const int32_t N = 64;
    for (int32_t tb = 0; tb < q_len; tb += bq) {
      RowTile t{0, tb, std::min(bq, q_len - tb), q_len, seq_len, 0};
      const int32_t end = kv_range_end(t);
      const int32_t minl = kv_min_limit(t);
      int32_t brute_end = 0;
      for (int32_t i = tb; i < tb + t.num_tok; ++i) {
        for (int32_t j = 0; j < seq_len; ++j) {
          const bool vis = j <= seq_len - q_len + i;  // the definition
          if (vis != (j <= kv_limit(q_len, seq_len, i))) ++bad;
          if (vis) brute_end = std::max(brute_end, j + 1);
          if (vis != (j <= i)) topleft_differs = true;
        }
      }
      if (end != brute_end) ++bad;
      for (int32_t tile = 0; tile < num_kv_tiles(end, N); ++tile) {
        bool any_masked = false;
        for (int32_t i = tb; i < tb + t.num_tok; ++i) {
          for (int32_t j = tile * N; j < tile * N + N; ++j) {
            if (j > kv_limit(q_len, seq_len, i)) any_masked = true;
          }
        }
        if (tile_needs_mask(tile, N, minl) != any_masked) ++bad;
      }
      if (num_kv_tiles(end, N) * N < end ||
          (num_kv_tiles(end, N) - 1) * N >= end) {
        ++bad;
      }
    }
  }
  check(bad == 0, "causal limits, range end and exact mask predicate");
  check(topleft_differs, "negative control: top-left alignment differs");
}

// Property: kv_window_lo/hi give exactly vLLM's visible set
// (causal j <= p, non-causal j < seq_len; window W > 0 also |p - j| < W) for
// every (row, position); the CTA range [begin, end) is the hull of its rows'
// sets; the tiles start at the one holding begin and the split partition
// covers them contiguously; tile_needs_window_mask and
// subtile_needs_window_mask are EXACT (true iff some real row has a
// position of the (sub-)tile outside its window); outside_window is the
// per-score predicate. With W = 0 and causal everything equals the causal
// geometry. Why: an unmasked boundary tile leaks a token outside the window,
// a dropped first tile loses the window's oldest tokens. Negative control:
// lo = p - W (one position too many) disagrees with the definition.
void test_window() {
  std::printf("test_window\n");
  std::mt19937_64 rng(13);
  int bad = 0;
  bool off_by_one_differs = false;
  const int32_t N = 64;
  for (int it = 0; it < 2000; ++it) {
    const int32_t q_len = 1 + static_cast<int32_t>(rng() % 50);
    const int32_t seq_len = q_len + static_cast<int32_t>(rng() % 400);
    const int32_t bq = 1 + static_cast<int32_t>(rng() % 16);
    // W = 0 (no window) a quarter of the time, else 1 .. seq_len + 70 (a
    // window wider than the sequence included).
    const int32_t window =
        rng() % 4 == 0 ? 0 : 1 + static_cast<int32_t>(rng() % (seq_len + 70));
    const bool causal = rng() % 2 == 0;
    const int32_t sub = rng() % 2 == 0 ? 16 : 32;
    const int32_t S = 1 + static_cast<int32_t>(rng() % 5);
    for (int32_t tb = 0; tb < q_len; tb += bq) {
      const RowTile t{0, tb, std::min(bq, q_len - tb), q_len, seq_len, 0};
      const int32_t begin = kv_window_range_begin(t, window);
      const int32_t end = kv_window_range_end(t, window, causal);
      const int32_t max_lo = kv_window_max_lo(t, window);
      const int32_t min_hi = kv_window_min_hi(t, window, causal);
      std::vector<int32_t> lo(static_cast<size_t>(t.num_tok));
      std::vector<int32_t> hi(static_cast<size_t>(t.num_tok));
      int32_t brute_begin = INT32_MAX, brute_end = 0;
      for (int32_t k = 0; k < t.num_tok; ++k) {
        const int32_t i = tb + k;
        const int32_t p = seq_len - q_len + i;
        lo[static_cast<size_t>(k)] = kv_window_lo(q_len, seq_len, i, window);
        hi[static_cast<size_t>(k)] =
            kv_window_hi(q_len, seq_len, i, window, causal);
        const int32_t lo_ctrl = window > 0 ? std::max(0, p - window) : 0;
        for (int32_t j = 0; j < seq_len; ++j) {
          const bool vis = (causal ? j <= p : j < seq_len) &&
                           (window == 0 || std::abs(p - j) < window);
          const bool impl = !outside_window(j, lo[static_cast<size_t>(k)],
                                            hi[static_cast<size_t>(k)]);
          if (vis != impl) ++bad;
          if (vis != (j >= lo_ctrl && j <= hi[static_cast<size_t>(k)])) {
            off_by_one_differs = true;
          }
          if (vis) {
            brute_begin = std::min(brute_begin, j);
            brute_end = std::max(brute_end, j + 1);
          }
        }
      }
      if (begin != brute_begin || end != brute_end) ++bad;
      if (window == 0 && causal &&
          (begin != 0 || end != kv_range_end(t) || min_hi != kv_min_limit(t))) {
        ++bad;
      }
      const int32_t first = kv_first_tile(begin, N);
      const int32_t last = num_kv_tiles(end, N);
      if (first * N > begin || (first + 1) * N <= begin) ++bad;
      int32_t prev = first;
      for (int32_t s = 0; s < S; ++s) {
        int32_t b = 0, e = 0;
        window_split_tile_range(s, S, begin, end, N, b, e);
        if (b != prev || e < b) ++bad;
        prev = e;
      }
      if (prev != last) ++bad;
      auto any_outside = [&](int32_t j0, int32_t n) {
        for (int32_t k = 0; k < t.num_tok; ++k) {
          for (int32_t j = j0; j < j0 + n; ++j) {
            if (outside_window(j, lo[static_cast<size_t>(k)],
                               hi[static_cast<size_t>(k)])) {
              return true;
            }
          }
        }
        return false;
      };
      for (int32_t tile = first; tile < last; ++tile) {
        if (tile_needs_window_mask(tile, N, max_lo, min_hi) !=
            any_outside(tile * N, N)) {
          ++bad;
        }
        for (int32_t tok0 = tile * N; tok0 < tile * N + N; tok0 += sub) {
          if (subtile_needs_window_mask(tok0, sub, max_lo, min_hi) !=
              any_outside(tok0, sub)) {
            ++bad;
          }
        }
      }
    }
  }
  check(bad == 0,
        "window bounds, range, tile range, splits and exact mask predicates");
  check(off_by_one_differs, "negative control: lo = p - W differs");
}

}  // namespace

// ================================================ ctypes API (Python tests)
extern "C" {

// Gathers every (request, position, KV head) of a paged cache through the
// header's address math, from the views' data pointers and element strides
// exactly as launch.cu will receive them. Output rows are request-major,
// token-major: out_k[(tok, head, d)] over all requests concatenated.
// widths[4] receives the per-stream cp.async widths. bounds8 holds, per
// stream, the [begin, end) byte range of the storage behind its view: an
// address outside it is reported (return -2) instead of read, so a wrong
// address fails the test cleanly instead of crashing the interpreter.
// Returns 0, -1 if the block size is invalid, -2 on an out-of-range address.
int geom_gather(const void* k_base, const void* v_base, const void* ks_base,
                const void* vs_base, const int64_t* sizes3,
                const int64_t* k_strides3, const int64_t* v_strides3,
                const int64_t* ks_strides3, const int64_t* vs_strides3,
                const uint64_t* bounds8, const int32_t* block_table,
                int64_t bt_row_stride, const int32_t* seq_lens,
                int32_t num_reqs, int32_t D, int32_t block_size, int8_t* out_k,
                int8_t* out_v, float* out_ks, float* out_vs, int32_t* widths) {
  KernelParams p{};
  p.stream[kStreamK] = make_stream(k_base, sizes3, k_strides3, 1, D);
  p.stream[kStreamV] = make_stream(v_base, sizes3, v_strides3, 1, D);
  p.stream[kStreamKScale] = make_stream(ks_base, sizes3, ks_strides3, 4, 4);
  p.stream[kStreamVScale] = make_stream(vs_base, sizes3, vs_strides3, 4, 4);
  p.block_table = block_table;
  p.bt_row_stride = bt_row_stride;
  p.block_size = FastDivmod::make(static_cast<uint32_t>(block_size));
  if (!p.block_size.valid()) return -1;
  for (int s = 0; s < kNumStreams; ++s) widths[s] = p.stream[s].width;
  void* outs[kNumStreams] = {out_k, out_v, out_ks, out_vs};
  const int32_t H = static_cast<int32_t>(sizes3[2]);
  int64_t row = 0;
  for (int32_t r = 0; r < num_reqs; ++r) {
    for (int32_t pos = 0; pos < seq_lens[r]; ++pos, ++row) {
      uint32_t slot = 0;
      const int64_t page = token_page(p, r, static_cast<uint32_t>(pos), slot);
      for (int32_t h = 0; h < H; ++h) {
        const int64_t o = (row * H + h);
        for (int s = 0; s < kNumStreams; ++s) {
          const StreamGeometry& g = p.stream[s];
          const uint64_t a =
              reinterpret_cast<uint64_t>(g.address(page, slot, h));
          if (a < bounds8[2 * s] ||
              a + static_cast<uint64_t>(g.row_bytes) > bounds8[2 * s + 1]) {
            return -2;
          }
          std::memcpy(static_cast<char*>(outs[s]) + o * g.row_bytes,
                      reinterpret_cast<const void*>(a),
                      static_cast<size_t>(g.row_bytes));
        }
      }
    }
  }
  return 0;
}

void geom_split_tile_range(int32_t split, int32_t num_splits, int32_t num_tiles,
                           int32_t* begin, int32_t* end) {
  split_tile_range(split, num_splits, num_tiles, *begin, *end);
}

int32_t geom_bq_capacity(int32_t wm, int32_t G) { return bq_capacity(wm, G); }

// out6 = {req, tok_begin, num_tok, q_len, seq_len, q_start}; returns 1 if the
// tile has work, else 0.
int32_t geom_row_tile(const int32_t* qsl, const int32_t* seq_lens,
                      int32_t num_reqs, int32_t bq, int32_t x, int32_t* out6) {
  RowTile t{};
  const bool ok =
      row_tile(qsl, seq_lens, FastDivmod::make(static_cast<uint32_t>(bq)), bq,
               num_reqs, x, t);
  if (!ok) return 0;
  const int32_t v[6] = {t.req,   t.tok_begin, t.num_tok,
                        t.q_len, t.seq_len,   t.q_start};
  std::memcpy(out6, v, sizeof v);
  return 1;
}

int32_t geom_kv_limit(int32_t q_len, int32_t seq_len, int32_t tok) {
  return kv_limit(q_len, seq_len, tok);
}

// {range_end, min_limit} of a row tile.
void geom_tile_range(int32_t q_len, int32_t seq_len, int32_t tok_begin,
                     int32_t num_tok, int32_t* out2) {
  const RowTile t{0, tok_begin, num_tok, q_len, seq_len, 0};
  out2[0] = kv_range_end(t);
  out2[1] = kv_min_limit(t);
}

int32_t geom_tile_needs_mask(int32_t tile, int32_t tile_tokens,
                             int32_t min_limit) {
  return tile_needs_mask(tile, tile_tokens, min_limit) ? 1 : 0;
}

// {lo, hi} of query token `tok`.
void geom_kv_window(int32_t q_len, int32_t seq_len, int32_t tok, int32_t window,
                    int32_t causal, int32_t* out2) {
  out2[0] = kv_window_lo(q_len, seq_len, tok, window);
  out2[1] = kv_window_hi(q_len, seq_len, tok, window, causal != 0);
}

// {range_begin, range_end, max_lo, min_hi} of a windowed row tile.
void geom_window_tile_range(int32_t q_len, int32_t seq_len, int32_t tok_begin,
                            int32_t num_tok, int32_t window, int32_t causal,
                            int32_t* out4) {
  const RowTile t{0, tok_begin, num_tok, q_len, seq_len, 0};
  out4[0] = kv_window_range_begin(t, window);
  out4[1] = kv_window_range_end(t, window, causal != 0);
  out4[2] = kv_window_max_lo(t, window);
  out4[3] = kv_window_min_hi(t, window, causal != 0);
}

void geom_window_split_tile_range(int32_t split, int32_t num_splits,
                                  int32_t range_begin, int32_t range_end,
                                  int32_t tile_tokens, int32_t* begin,
                                  int32_t* end) {
  window_split_tile_range(split, num_splits, range_begin, range_end,
                          tile_tokens, *begin, *end);
}

}  // extern "C"

// ================================================= device compile probe
// Under nvcc every function above is instantiated in device code, so a
// construct that only compiles on the host fails the build (run_geometry.sh).
#if defined(__CUDACC__)
__global__ void geometry_device_probe(KernelParams p, int64_t* out) {
  uint32_t slot = 0;
  const int64_t page = token_page(p, 0, threadIdx.x, slot);
  int64_t acc = p.stream[kStreamK].offset(page, slot, 0) +
                (p.stream[kStreamVScale].address(page, slot, 1) -
                 p.stream[kStreamVScale].base);
  RowTile t{};
  if (row_tile(p.query_start_loc, p.seq_lens, p.bq_div, p.bq, p.num_reqs,
               blockIdx.x, t)) {
    int32_t b = 0, e = 0;
    split_tile_range(blockIdx.z, p.num_splits,
                     num_kv_tiles(kv_range_end(t), 64), b, e);
    int32_t tok = 0, g = 0;
    acc += b + e + tile_needs_mask(b, 64, kv_min_limit(t)) +
           row_to_token_head(threadIdx.x, p.group, t.num_tok, tok, g) + tok +
           g + bq_capacity(p.wm, p.group_size);
    acc += q_offset(p, t.q_start, g) + out_offset(p, t.q_start, g) +
           ws_o_offset(p, 0, t.q_start, g) + ws_lse_offset(p, 0, t.q_start, g);
    // Windowed geometry with run-time window and causal flag.
    const int32_t w = p.max_query_len;
    const bool causal = (p.wm & 1) != 0;
    window_split_tile_range(blockIdx.z, p.num_splits,
                            kv_window_range_begin(t, w),
                            kv_window_range_end(t, w, causal), 64, b, e);
    acc += b + e +
           tile_needs_window_mask(b, 64, kv_window_max_lo(t, w),
                                  kv_window_min_hi(t, w, causal)) +
           subtile_needs_window_mask(b * 64, 16, kv_window_max_lo(t, w),
                                     kv_window_min_hi(t, w, causal)) +
           outside_window(static_cast<int32_t>(threadIdx.x),
                          kv_window_lo(1, 2, 0, w),
                          kv_window_hi(1, 2, 0, w, causal));
  }
  out[threadIdx.x] = acc;
}
#endif

#ifndef PTH_GEOMETRY_SHARED
int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);  // keep progress if a mutant crashes
  test_fastdivmod();
  test_stream_widths();
  test_stride_fuzz();
  test_int64_offsets();
  test_token_page();
  test_split_partition();
  test_gqa_packing();
  test_row_tiles();
  test_causal();
  test_window();
  std::printf("test_geometry: %d checks, %d failed\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
#endif
