// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// test_primitives.cu: one source, two uses.
//
// 1. Host mode (g++ -x c++): evaluates the __host__ __device__ bit and index
//    math of dequant.cuh and smem_layout.cuh for every input that matters and
//    prints it as JSON with the same keys as the simulator's golden tables
//    (tests/sim/ptxsim/golden.py). tests/cpu/test_primitives_golden.py
//    compares the two key by key and fails on any difference. Inputs are
//    generated here, independently of the golden file, in the same
//    deterministic order. A few properties that need no golden table (the
//    shared-memory totals, pi round trip) are static_asserts.
//
// 2. Device mode (nvcc -cubin, sm_80/86/89): the probe kernels at the end use
//    every ptx.cuh wrapper, so tools/ptxas_gate.py (0 spills) and
//    tools/sass_check.py (LDGSTS, LDSM, HMMA present; no I2F) can check the
//    real instructions, and the per-kernel SASS opcode counts show what cvt4
//    costs. The kernels are not run here (no GPU in CPU checks).
//
// Build and run: tests/cpp/run_primitives.sh host OUT_DIR [INCLUDE_DIR].

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <iterator>
#include <vector>

#include "dequant.cuh"
#include "ptx.cuh"
#include "smem_layout.cuh"

namespace {

using pth_attn::SmemLayout;
using pth_attn::StageLayout;

// The golden tables use tile N = 64 and 64 Q rows (WM = 4).
constexpr int kTileN = 64;
constexpr int kQRowsMax = 64;
constexpr int kRowsPerWarp = 16;
constexpr int kStages = 2;
constexpr int kLanes = 32;

// Tile table: total shared memory per (D, WM) and stage bytes.
static_assert(SmemLayout<256, kTileN, 4 * kRowsPerWarp, kStages>::kBytes ==
              99328);
static_assert(SmemLayout<256, kTileN, 2 * kRowsPerWarp, kStages>::kBytes ==
              82944);
static_assert(SmemLayout<256, kTileN, 1 * kRowsPerWarp, kStages>::kBytes ==
              74752);
static_assert(SmemLayout<128, kTileN, 4 * kRowsPerWarp, kStages>::kBytes ==
              50176);
static_assert(SmemLayout<64, kTileN, 4 * kRowsPerWarp, kStages>::kBytes ==
              25600);
static_assert(StageLayout<256, kTileN>::kBytes == 33280);
static_assert(StageLayout<128, kTileN>::kBytes == 16896);
static_assert(StageLayout<64, kTileN>::kBytes == 8704);

constexpr bool pi_round_trips() {
  for (int k = 0; k < pth_attn::kPiChunk; ++k) {
    if (pth_attn::pi_d_to_k(pth_attn::pi_k_to_d(k)) != k) return false;
  }
  return true;
}
static_assert(pi_round_trips(), "pi_d_to_k must invert pi_k_to_d");

// ---------------------------------------------------------------------------
// Minimal JSON writer (objects of arrays of integers and strings)
// ---------------------------------------------------------------------------
class Json {
 public:
  void open() {
    sep();
    out_ += '{';
    first_ = true;
  }
  void close() {
    out_ += '}';
    first_ = false;
  }
  void key(const std::string& k) {
    sep();
    out_ += '"' + k + "\":";
    first_ = true;  // the value follows without a comma
  }
  void num(int64_t v) {
    sep();
    out_ += std::to_string(v);
  }
  void str(const std::string& s) {
    sep();
    out_ += '"' + s + '"';
  }
  void arr_open() {
    sep();
    out_ += '[';
    first_ = true;
  }
  void arr_close() {
    out_ += ']';
    first_ = false;
  }
  template <typename T>
  void list(const std::vector<T>& v) {
    arr_open();
    for (const T& x : v) num(static_cast<int64_t>(x));
    arr_close();
  }
  const std::string& text() const { return out_; }

 private:
  void sep() {
    if (!first_) out_ += ',';
    first_ = false;
  }
  std::string out_;
  bool first_ = true;
};

std::vector<int> int8_range() {
  std::vector<int> x;
  for (int v = -128; v < 128; ++v) x.push_back(v);
  return x;
}

uint32_t word_from_bytes(int b0, int b1, int b2, int b3) {
  return static_cast<uint32_t>(static_cast<uint8_t>(b0)) |
         static_cast<uint32_t>(static_cast<uint8_t>(b1)) << 8 |
         static_cast<uint32_t>(static_cast<uint8_t>(b2)) << 16 |
         static_cast<uint32_t>(static_cast<uint8_t>(b3)) << 24;
}

void emit_magic(Json& j) {
  const std::vector<int> x = int8_range();
  std::vector<uint32_t> words, lo, hi, f16_bits;
  for (int v : x) words.push_back(word_from_bytes(v, v, v, v));
  for (size_t i = 0; i < x.size(); i += 4) {
    words.push_back(word_from_bytes(x[i], x[i + 1], x[i + 2], x[i + 3]));
  }
  for (uint32_t w : words) {
    const pth_attn::F16x2Pair p = pth_attn::cvt4_s8_f16x2(w);
    lo.push_back(p.lo);
    hi.push_back(p.hi);
  }
  for (size_t i = 0; i < x.size(); ++i) f16_bits.push_back(lo[i] & 0xFFFF);
  j.key("magic");
  j.open();
  j.key("sign_flip");
  j.num(pth_attn::kMagicSignFlip);
  j.key("exp_bytes");
  j.num(pth_attn::kMagicExpBytes);
  j.key("bias_half2");
  j.num(pth_attn::kMagicBiasHalf2);
  j.key("sel_bytes_01");
  j.num(pth_attn::kSelBytes01);
  j.key("sel_bytes_23");
  j.num(pth_attn::kSelBytes23);
  j.key("cvt4");
  j.open();
  j.key("int8");
  j.list(x);
  j.key("f16_bits");
  j.list(f16_bits);
  j.key("words");
  j.list(words);
  j.key("lo");
  j.list(lo);
  j.key("hi");
  j.list(hi);
  j.close();
  j.close();
}

void emit_v(Json& j) {
  j.key("v_selectors");
  j.open();
  j.key("d01");
  j.num(pth_attn::kVSelD01);
  j.key("d23");
  j.num(pth_attn::kVSelD23);
  j.close();
  // 16 quadruples of words whose 16 bytes are 16 consecutive int8 values.
  constexpr int kQuads = 16;
  j.key("v_b_fragments");
  j.open();
  j.key("words");
  j.arr_open();
  std::vector<std::vector<uint32_t>> all;
  for (int qd = 0; qd < kQuads; ++qd) {
    std::vector<uint32_t> w;
    for (int i = 0; i < 4; ++i) {
      const int b = -128 + 16 * qd + 4 * i;
      w.push_back(word_from_bytes(b, b + 1, b + 2, b + 3));
    }
    j.list(w);
    all.push_back(w);
  }
  j.arr_close();
  j.key("regs");
  j.arr_open();
  for (const auto& w : all) {
    const uint32_t in[4] = {w[0], w[1], w[2], w[3]};
    uint32_t b[4][2];
    pth_attn::v_b_fragments(in, b);
    j.arr_open();
    for (auto& nt : b) j.list(std::vector<uint32_t>{nt[0], nt[1]});
    j.arr_close();
  }
  j.arr_close();
  j.close();
}

std::string hex16(const std::vector<uint16_t>& v) {
  std::string s;
  char buf[5];
  for (uint16_t x : v) {
    std::snprintf(buf, sizeof(buf), "%04x", x);
    s += buf;
  }
  return s;
}

// Prescale of every finite bf16 / fp16 value x: with its own e (x is the row
// max) and with each fixed e. Both halves of the f16x2 are checked equal.
bool emit_prescale(Json& j) {
  const int fixed_e[] = {-112, -60, -17, 0, 1, 20, 60, 113};
  bool ok = true;
  std::vector<float> xs[2];
  for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
    const float x = pth_attn::u32_as_f32(bits << 16);  // bf16 = top half of f32
    if (std::isfinite(x)) xs[0].push_back(x);
    const double h =
        pth_attn::host_emu::f16_bits_to_double(static_cast<uint16_t>(bits));
    if (std::isfinite(h)) xs[1].push_back(static_cast<float>(h));
  }
  const char* names[2] = {"bf16", "fp16"};
  j.key("prescale");
  j.open();
  j.key("fixed_e");
  j.list(std::vector<int>(std::begin(fixed_e), std::end(fixed_e)));
  for (int t = 0; t < 2; ++t) {
    auto run = [&](auto e_of) {
      std::vector<uint16_t> r;
      for (float x : xs[t]) {
        const uint32_t p = pth_attn::q_prescale_pack(x, x, e_of(x));
        if ((p & 0xFFFF) != (p >> 16)) ok = false;
        r.push_back(static_cast<uint16_t>(p & 0xFFFF));
      }
      return hex16(r);
    };
    std::vector<int> e_own;
    for (float x : xs[t])
      e_own.push_back(pth_attn::q_prescale_exponent(std::fabs(x)));
    j.key(names[t]);
    j.open();
    j.key("e_own");
    j.list(e_own);
    j.key("own");
    j.str(run(
        [](float x) { return pth_attn::q_prescale_exponent(std::fabs(x)); }));
    for (int e : fixed_e) {
      j.key(std::to_string(e));
      j.str(run([e](float) { return e; }));
    }
    j.close();
  }
  j.close();
  return ok;
}

template <int D>
void emit_per_d(Json& j) {
  using St = StageLayout<D, kTileN>;
  j.key(std::to_string(D));
  j.open();
  j.key("stage_offsets");
  j.open();
  j.key("K");
  j.num(St::kK);
  j.key("V");
  j.num(St::kV);
  j.key("ks");
  j.num(St::kKs);
  j.key("vs");
  j.num(St::kVs);
  j.key("bytes");
  j.num(St::kBytes);
  j.close();

  auto chunks = [&](const char* name, auto fn, int rows, int nchunks) {
    j.key(name);
    j.arr_open();
    for (int r = 0; r < rows; ++r) {
      std::vector<uint32_t> v;
      for (int c = 0; c < nchunks; ++c)
        v.push_back(fn(r, c) / pth_attn::kSmemChunkBytes);
      j.list(v);
    }
    j.arr_close();
  };
  const uint32_t chunk = pth_attn::kSmemChunkBytes;
  chunks(
      "k_chunk", [&](int t, int c) { return pth_attn::k_swz(t, chunk * c, D); },
      kTileN, D / 16);
  chunks(
      "v_chunk", [&](int t, int c) { return pth_attn::v_swz(t, chunk * c, D); },
      kTileN, D / 16);
  // Q chunk qc holds 8 fp16 elements: element index 8 * qc.
  chunks(
      "q_chunk", [&](int r, int c) { return pth_attn::q_swz(r, 8 * c, D); },
      kQRowsMax, 2 * D / 16);

  j.key("q_elem_offset");
  j.arr_open();
  for (int r = 0; r < kQRowsMax; ++r) {
    std::vector<uint32_t> v;
    for (int d = 0; d < D; ++d) v.push_back(pth_attn::q_elem_offset(r, d, D));
    j.list(v);
  }
  j.arr_close();

  j.key("k_ldsm");
  j.arr_open();
  for (int tok0 = 0; tok0 < kTileN; tok0 += 32) {
    j.arr_open();
    for (int s = 0; s < D / 16; ++s) {
      std::vector<uint32_t> v;
      for (int l = 0; l < kLanes; ++l)
        v.push_back(pth_attn::k_ldsm_x4_offset(l, tok0, s, D));
      j.list(v);
    }
    j.arr_close();
  }
  j.arr_close();

  j.key("q_ldsm");
  j.arr_open();
  for (int row0 = 0; row0 < kQRowsMax; row0 += kRowsPerWarp) {
    j.arr_open();
    for (int s = 0; s < D / 16; ++s) {
      std::vector<uint32_t> v;
      for (int l = 0; l < kLanes; ++l)
        v.push_back(pth_attn::q_ldsm_x4_offset(l, row0, s, D));
      j.list(v);
    }
    j.arr_close();
  }
  j.arr_close();

  j.key("v_lds32");
  j.arr_open();
  for (int kt = 0; kt < kTileN / 16; ++kt) {
    j.arr_open();
    for (int a = 0; a < D / 32; ++a) {
      j.arr_open();
      for (int i = 0; i < 4; ++i) {
        std::vector<uint32_t> v;
        for (int l = 0; l < kLanes; ++l) {
          v.push_back(pth_attn::v_lds32_offset(l, 16 * kt, a, i, D));
        }
        j.list(v);
      }
      j.arr_close();
    }
    j.arr_close();
  }
  j.arr_close();

  j.key("v_ntile_col_to_d");
  j.arr_open();
  for (int nt = 0; nt < D / 8; ++nt) {
    std::vector<int> v;
    for (int n = 0; n < 8; ++n) v.push_back(pth_attn::v_ntile_col_to_d(nt, n));
    j.list(v);
  }
  j.arr_close();

  // Register order of the epilogue: c0 of n-tiles 4a..4a+3, then c1.
  j.key("epilogue_columns");
  j.arr_open();
  for (int l = 0; l < kLanes; ++l) {
    j.arr_open();
    for (int a = 0; a < D / 32; ++a) {
      std::vector<int> v;
      for (int ci = 0; ci < 2; ++ci) {
        for (int b = 0; b < 4; ++b)
          v.push_back(pth_attn::pv_out_col(l, 4 * a + b, ci));
      }
      j.list(v);
    }
    j.arr_close();
  }
  j.arr_close();
  j.close();
}

// Checks without a golden table: epilogue_d0 is the first of the 8
// contiguous columns pv_out_col yields.
bool check_epilogue_d0() {
  for (int l = 0; l < kLanes; ++l) {
    for (int a = 0; a < 256 / 32; ++a) {
      int expect = pth_attn::epilogue_d0(l, a);
      for (int ci = 0; ci < 2; ++ci) {
        for (int b = 0; b < 4; ++b) {
          if (pth_attn::pv_out_col(l, 4 * a + b, ci) != expect++) return false;
        }
      }
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Json j;
  j.open();
  j.key("pi");
  {
    std::vector<int> pi, inv;
    for (int k = 0; k < pth_attn::kPiChunk; ++k)
      pi.push_back(pth_attn::pi_k_to_d(k));
    for (int d = 0; d < pth_attn::kPiChunk; ++d)
      inv.push_back(pth_attn::pi_d_to_k(d));
    j.list(pi);
    j.key("pi_inverse");
    j.list(inv);
  }
  emit_magic(j);
  emit_v(j);
  const bool halves_ok = emit_prescale(j);
  j.key("per_d");
  j.open();
  emit_per_d<64>(j);
  emit_per_d<128>(j);
  emit_per_d<256>(j);
  j.close();
  j.close();

  const bool epi_ok = check_epilogue_d0();
  FILE* f = argc > 1 ? std::fopen(argv[1], "w") : stdout;
  if (f == nullptr) {
    std::perror("open output");
    return 2;
  }
  std::fputs(j.text().c_str(), f);
  std::fputc('\n', f);
  if (f != stdout) std::fclose(f);
  if (!halves_ok) std::fprintf(stderr, "FAIL: q_prescale_pack halves differ\n");
  if (!epi_ok)
    std::fprintf(stderr, "FAIL: epilogue_d0 disagrees with pv_out_col\n");
  return halves_ok && epi_ok ? 0 : 1;
}

#if defined(__CUDACC__)
// ---------------------------------------------------------------------------
// Device probes (compiled with nvcc -cubin; never launched in CPU checks)
// ---------------------------------------------------------------------------
namespace probe {

using namespace pth_attn;

constexpr int kD = 64;
using Layout = SmemLayout<kD, kTileN, kRowsPerWarp, 1>;
using Stage = Layout::Stage;

// Every ptx.cuh wrapper once, wired like one k16 step of the real inner
// loop at D = 64: cp.async (16, 8, 4 and zero-fill) into Q + one stage,
// ldmatrix.x4 of Q (A) and K (B via cvt4), QK mma, S -> P packing, V LDS.32 +
// pairing + cvt4, PV mma, ldmatrix .trans, and the Q prescale helpers.
__global__ void __launch_bounds__(kLanes)
    probe_primitives(const uint8_t* __restrict__ g,
                     const float* __restrict__ qf, float* __restrict__ out) {
  __shared__ __align__(128) uint8_t smem[Layout::kBytes];
  const uint32_t lane = threadIdx.x;
  const uint32_t base = smem_u32(smem);
  constexpr int kTail = 16;  // last 16 B: filled by the 8-, 4- and zfill copies
  for (int i = static_cast<int>(lane); i < (Layout::kBytes - kTail) / 16;
       i += kLanes) {
    cp_async<16>(base + 16 * i, g + 16 * i);
  }
  const int tail = Layout::kBytes - kTail;
  if (lane == 0) cp_async<8>(base + tail, g + tail);
  if (lane == 1) cp_async<4>(base + tail + 8, g + tail + 8);
  if (lane == 2) cp_async_zfill<4>(base + tail + 12, g + tail + 12, 0);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  uint32_t a[4], kr[4], qt[4];
  ldmatrix_x4(a, base + Layout::kQ + q_ldsm_x4_offset(lane, 0, 0, kD));
  ldmatrix_x4(kr, base + Layout::stage_offset(0) + Stage::kK +
                      k_ldsm_x4_offset(lane, 0, 0, kD));
  ldmatrix_x4<true>(qt, base + Layout::kQ + q_ldsm_x4_offset(lane, 0, 1, kD));
  float s[4][4] = {};
  for (int m = 0; m < 4; ++m) {
    const F16x2Pair p = cvt4_s8_f16x2(kr[m]);
    const uint32_t b[2] = {p.lo, p.hi};
    mma_m16n8k16_f16f32(s[m], a, b);
  }
  const uint32_t bt[2] = {qt[0], qt[1]};
  mma_m16n8k16_f16f32(s[0], qt, bt);

  // P (A operand of PV) from two QK accumulator n-tiles (FA2 register reuse).
  const uint32_t p[4] = {
      pack_f16x2_rn(s[0][0], s[0][1]), pack_f16x2_rn(s[0][2], s[0][3]),
      pack_f16x2_rn(s[1][0], s[1][1]), pack_f16x2_rn(s[1][2], s[1][3])};
  uint32_t w[4];
  for (int i = 0; i < 4; ++i) {
    w[i] = *reinterpret_cast<const uint32_t*>(
        smem + Layout::stage_offset(0) + Stage::kV +
        v_lds32_offset(lane, 0, 0, i, kD));
  }
  uint32_t vb[4][2];
  v_b_fragments(w, vb);
  float o[4][4] = {};
  for (int n = 0; n < 4; ++n) mma_m16n8k16_f16f32(o[n], p, vb[n]);

  const int e = q_prescale_exponent(qf[0]);
  const uint32_t qq = q_prescale_pack(qf[1 + lane], qf[1 + kLanes + lane], e);
  float acc = u32_as_f32(qq);
  for (auto& on : o) {
    for (float x : on) acc += x;
  }
  // s[2], s[3] feed the next k16 step in the real kernel; use them so the
  // compiler keeps all four QK MMAs.
  for (int m = 2; m < 4; ++m) {
    for (float x : s[m]) acc += x;
  }
  out[lane] = acc;
}

// Counting kernels: one primitive each, no indexing, so the SASS opcode
// counts are the primitive's own (test_primitives_golden.py reports them).
__global__ void probe_cvt4(const uint32_t* __restrict__ in,
                           uint32_t* __restrict__ out) {
  const F16x2Pair p = cvt4_s8_f16x2(in[0]);
  out[0] = p.lo;
  out[1] = p.hi;
}

__global__ void probe_cvt4_x4(const uint4* __restrict__ in,
                              uint4* __restrict__ out) {
  const uint4 w = in[0];
  const F16x2Pair p0 = cvt4_s8_f16x2(w.x), p1 = cvt4_s8_f16x2(w.y),
                  p2 = cvt4_s8_f16x2(w.z), p3 = cvt4_s8_f16x2(w.w);
  out[0] = make_uint4(p0.lo, p0.hi, p1.lo, p1.hi);
  out[1] = make_uint4(p2.lo, p2.hi, p3.lo, p3.hi);
}

__global__ void probe_v_b_fragments(const uint4* __restrict__ in,
                                    uint4* __restrict__ out) {
  const uint4 v = in[0];
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  uint32_t b[4][2];
  v_b_fragments(w, b);
  out[0] = make_uint4(b[0][0], b[0][1], b[1][0], b[1][1]);
  out[1] = make_uint4(b[2][0], b[2][1], b[3][0], b[3][1]);
}

__global__ void probe_q_prescale(const float* __restrict__ in,
                                 uint32_t* __restrict__ out) {
  out[0] = q_prescale_pack(in[1], in[2], q_prescale_exponent(in[0]));
}

}  // namespace probe
#endif  // __CUDACC__
