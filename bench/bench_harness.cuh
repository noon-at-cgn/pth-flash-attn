// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// bench_harness.cuh: host-side plumbing shared by the two hardware-ceiling
// programs, sol_copy.cu and mma_ceiling.cu. Device code lives in those files;
// this header only holds what both need on the host.
//
// Pieces and why they exist:
//   Args        --key value parser. No env vars anywhere in the bench: a run is
//               fully described by its argv, and the argv goes into the JSON.
//   mono_ns()   CLOCK_MONOTONIC in ns. bench/nvml_log.py stamps its samples
//               with Python's time.monotonic_ns(), which reads the same clock
//               on Linux; docker does not namespace CLOCK_MONOTONIC, so a
//               block's [t_start, t_end] can be matched to NVML samples taken
//               by a logger running in a different container (bench/stats.py
//               does that).
//   KernelTimer per-kernel GPU durations from the CUPTI Activity API (kernel
//               start and end timestamps, no launch gaps, no host time). This
//               is the default timer. FlashInfer's
//               bench_gpu_time_with_cupti is not used: it needs the
//               `cupti-python` package and, when that is missing, silently
//               falls back to CUDA events. The C API (libcupti.so.13, shipped
//               with the CUDA 13 toolkit and the nvidia-cuda-cupti wheel) is
//               always available, so it is called directly.
//               --timer events selects CUDA events instead, and the JSON says
//               which timer produced the numbers.
//   Schedule    the mirrored ABBA order is produced by `python3 bench/stats.py
//               schedule` (single source); the programs only parse and validate
//               it.
//   Json        minimal streaming JSON writer for the raw per-process output
//               that bench/bench_report.py turns into schema-valid results.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace bench {

// ---------------------------------------------------------------- errors
// A failed CUDA/CUPTI call ends the process with exit code 2. A half-measured
// block must never be written as if it were valid, so there is no recovery
// path.
#define BENCH_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t bench_err_ = (expr);                                           \
    if (bench_err_ != cudaSuccess) {                                           \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                     \
                   cudaGetErrorString(bench_err_), __FILE__, __LINE__, #expr); \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

#define BENCH_CUPTI(expr)                                                     \
  do {                                                                        \
    CUptiResult bench_res_ = (expr);                                          \
    if (bench_res_ != CUPTI_SUCCESS) {                                        \
      const char* bench_msg_ = nullptr;                                       \
      cuptiGetResultString(bench_res_, &bench_msg_);                          \
      std::fprintf(stderr, "CUPTI error %s at %s:%d: %s\n",                   \
                   bench_msg_ ? bench_msg_ : "?", __FILE__, __LINE__, #expr); \
      std::exit(2);                                                           \
    }                                                                         \
  } while (0)

[[noreturn]] inline void die(const std::string& msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(64);  // EX_USAGE: bad arguments, nothing was measured
}

// ---------------------------------------------------------------- clocks
inline uint64_t mono_ns() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
         static_cast<uint64_t>(ts.tv_nsec);
}

inline uint64_t unix_ns() {
  timespec ts{};
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
         static_cast<uint64_t>(ts.tv_nsec);
}

// ---------------------------------------------------------------- arguments
class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 0; i < argc; ++i) argv_.emplace_back(argv[i]);
    for (int i = 1; i < argc; ++i) {
      std::string k = argv[i];
      if (k.rfind("--", 0) != 0) die("unexpected positional argument " + k);
      k = k.substr(2);
      // "--flag" alone is a boolean; "--key value" takes the next token unless
      // that token is itself an option.
      if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
        kv_[k] = argv[++i];
      } else {
        kv_[k] = "";
      }
    }
  }
  bool has(const std::string& k) const { return kv_.count(k) != 0; }
  std::string str(const std::string& k, const std::string& dflt) const {
    used_.push_back(k);
    auto it = kv_.find(k);
    return it == kv_.end() ? dflt : it->second;
  }
  std::string req(const std::string& k) const {
    used_.push_back(k);
    auto it = kv_.find(k);
    if (it == kv_.end() || it->second.empty()) die("--" + k + " is required");
    return it->second;
  }
  int64_t i64(const std::string& k, int64_t dflt) const {
    return has(k) ? parse_i64(k, str(k, "")) : (used_.push_back(k), dflt);
  }
  double f64(const std::string& k, double dflt) const {
    if (!has(k)) {
      used_.push_back(k);
      return dflt;
    }
    const std::string v = str(k, "");
    char* end = nullptr;
    errno = 0;
    const double x = std::strtod(v.c_str(), &end);
    if (errno != 0 || end == v.c_str() || *end != '\0')
      die("--" + k + " is not a number: " + v);
    return x;
  }
  // Unknown options are an error: a typo must not silently fall back to a
  // default and produce a run that is not the one the operator asked for.
  void reject_unknown() const {
    for (const auto& kv : kv_) {
      if (std::find(used_.begin(), used_.end(), kv.first) == used_.end()) {
        die("unknown option --" + kv.first);
      }
    }
  }
  const std::vector<std::string>& argv() const { return argv_; }

  static int64_t parse_i64(const std::string& k, const std::string& v) {
    char* end = nullptr;
    errno = 0;
    const long long x = std::strtoll(v.c_str(), &end, 10);
    if (errno != 0 || end == v.c_str() || *end != '\0')
      die("--" + k + " is not an integer: " + v);
    return static_cast<int64_t>(x);
  }

 private:
  std::vector<std::string> argv_;
  std::map<std::string, std::string> kv_;
  mutable std::vector<std::string> used_;
};

inline std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

// ---------------------------------------------------------------- schedule
// The schedule is a list of arm indices produced by `bench/stats.py schedule`.
// It is made of rounds; every round is a permutation of all arms, and
// consecutive rounds are mirror images (A B C | C B A | A B C ...), which
// cancels any drift that is linear in time. The round of position k
// is k / n_arms. Validation here is a guard against hand-written schedules.
inline std::vector<int> parse_schedule(const std::string& s, int n_arms) {
  std::vector<int> out;
  for (const auto& tok : split(s, ',')) {
    const int64_t a = Args::parse_i64("schedule", tok);
    if (a < 0 || a >= n_arms) die("schedule entry out of range: " + tok);
    out.push_back(static_cast<int>(a));
  }
  if (out.size() % static_cast<size_t>(n_arms) != 0) {
    die("schedule length is not a multiple of the number of arms");
  }
  for (size_t r = 0; r < out.size() / n_arms; ++r) {
    std::vector<int> seen(n_arms, 0);
    for (int j = 0; j < n_arms; ++j) seen[out[r * n_arms + j]]++;
    for (int a = 0; a < n_arms; ++a) {
      if (seen[a] != 1)
        die("schedule round " + std::to_string(r) +
            " is not a permutation of the arms");
    }
  }
  return out;
}

// ---------------------------------------------------------------- statistics
// on the host Only used for calibration and console progress; all reported
// statistics are computed by bench/stats.py from the raw durations.
inline double median_of(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------------------------------------------------------- CUPTI kernel
// timer CUPTI hands us activity buffers; 8 MiB holds well over 10^4 kernel
// records (CUpti_ActivityKernel10 is a few hundred bytes), so a 20-launch block
// never spills into a second buffer, and the 8-byte alignment is what the CUPTI
// documentation requires for activity buffers.
constexpr size_t kCuptiBufferBytes = 8u << 20;
constexpr size_t kCuptiBufferAlign = 8;

struct KernelRecord {
  uint64_t start_ns;
  uint64_t end_ns;
  std::string name;
};

class KernelTimer {
 public:
  enum class Kind { kCupti, kEvents };

  static KernelTimer& instance() {
    static KernelTimer t;
    return t;
  }

  // Must run before the first kernel we want timed (CUPTI attaches to the
  // context created by the first runtime call after enabling).
  void start(Kind kind) {
    kind_ = kind;
    if (kind_ != Kind::kCupti) return;
    BENCH_CUPTI(cuptiActivityRegisterCallbacks(&KernelTimer::on_request,
                                               &KernelTimer::on_complete));
    // CONCURRENT_KERNEL does not serialise kernels (the plain KERNEL kind
    // does), so the timed kernels run exactly as they would without the
    // profiler.
    BENCH_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  }

  Kind kind() const { return kind_; }
  const char* kind_name() const {
    return kind_ == Kind::kCupti ? "cupti" : "cuda_event";
  }

  // Call after the stream is synchronised. Returns every recorded kernel whose
  // (mangled) name contains `needle` ("" = all), in start order, and empties
  // the buffer so the next block starts clean.
  std::vector<KernelRecord> collect_records(const char* needle) {
    if (kind_ != Kind::kCupti) return {};
    BENCH_CUPTI(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    std::vector<KernelRecord> recs;
    {
      std::lock_guard<std::mutex> lk(mu_);
      recs.swap(records_);
    }
    std::sort(recs.begin(), recs.end(),
              [](const KernelRecord& a, const KernelRecord& b) {
                return a.start_ns < b.start_ns;
              });
    std::vector<KernelRecord> out;
    for (auto& r : recs) {
      if (r.name.find(needle) != std::string::npos) out.push_back(std::move(r));
    }
    return out;
  }

  // Durations (ns) of collect_records(needle).
  std::vector<uint64_t> collect(const char* needle) {
    std::vector<uint64_t> out;
    for (const auto& r : collect_records(needle))
      out.push_back(r.end_ns - r.start_ns);
    return out;
  }

  void stop() {
    if (kind_ != Kind::kCupti) return;
    BENCH_CUPTI(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    BENCH_CUPTI(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  }

 private:
  KernelTimer() = default;

  static void CUPTIAPI on_request(uint8_t** buffer, size_t* size,
                                  size_t* max_records) {
    *size = kCuptiBufferBytes;
    *buffer = static_cast<uint8_t*>(
        std::aligned_alloc(kCuptiBufferAlign, kCuptiBufferBytes));
    *max_records = 0;  // 0 = as many as fit
    if (*buffer == nullptr) {
      std::fprintf(stderr, "CUPTI buffer allocation failed\n");
      std::exit(2);
    }
  }

  static void CUPTIAPI on_complete(CUcontext, uint32_t, uint8_t* buffer, size_t,
                                   size_t valid) {
    KernelTimer& self = instance();
    CUpti_Activity* rec = nullptr;
    std::lock_guard<std::mutex> lk(self.mu_);
    while (cuptiActivityGetNextRecord(buffer, valid, &rec) == CUPTI_SUCCESS) {
      if (rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        const auto* k = reinterpret_cast<const CUpti_ActivityKernel10*>(rec);
        // The name pointer is owned by CUPTI; copy it now.
        self.records_.push_back({k->start, k->end, k->name ? k->name : ""});
      }
    }
    size_t dropped = 0;
    if (cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped) ==
            CUPTI_SUCCESS &&
        dropped) {
      // A dropped record would make a block look like it had fewer launches;
      // collect() callers check the count, but say why here.
      std::fprintf(stderr, "CUPTI dropped %zu records\n", dropped);
    }
    std::free(buffer);
  }

  Kind kind_ = Kind::kCupti;
  std::mutex mu_;
  std::vector<KernelRecord> records_;
};

inline KernelTimer::Kind parse_timer(const std::string& s) {
  if (s == "cupti") return KernelTimer::Kind::kCupti;
  if (s == "events") return KernelTimer::Kind::kEvents;
  die("--timer must be cupti or events");
}

// ---------------------------------------------------------------- JSON writer
// Streaming writer: begin/end objects and arrays, key/value pairs. Commas are
// tracked per nesting level. Strings are escaped for the characters our own
// identifiers and argv can contain.
class Json {
 public:
  explicit Json(FILE* f) : f_(f) { first_.push_back(true); }

  void begin_obj(const char* key = nullptr) { open(key, '{'); }
  void end_obj() { close('}'); }
  void begin_arr(const char* key = nullptr) { open(key, '['); }
  void end_arr() { close(']'); }

  void kv(const char* key, const std::string& v) {
    sep(key);
    str(v);
  }
  void kv(const char* key, const char* v) { kv(key, std::string(v)); }
  void kv(const char* key, bool v) {
    sep(key);
    std::fputs(v ? "true" : "false", f_);
  }
  void kv(const char* key, int v) {
    sep(key);
    std::fprintf(f_, "%d", v);
  }
  void kv(const char* key, int64_t v) {
    sep(key);
    std::fprintf(f_, "%lld", static_cast<long long>(v));
  }
  void kv(const char* key, uint64_t v) {
    sep(key);
    std::fprintf(f_, "%llu", static_cast<unsigned long long>(v));
  }
  void kv(const char* key, double v) {
    sep(key);
    std::fprintf(f_, "%.17g", v);
  }
  void val(uint64_t v) {
    sep(nullptr);
    std::fprintf(f_, "%llu", static_cast<unsigned long long>(v));
  }
  void val(double v) {
    sep(nullptr);
    std::fprintf(f_, "%.17g", v);
  }
  void val(const std::string& v) {
    sep(nullptr);
    str(v);
  }

 private:
  void sep(const char* key) {
    if (!first_.back()) std::fputc(',', f_);
    first_.back() = false;
    if (key) {
      str(key);
      std::fputc(':', f_);
    }
  }
  void open(const char* key, char c) {
    sep(key);
    std::fputc(c, f_);
    first_.push_back(true);
  }
  void close(char c) {
    first_.pop_back();
    std::fputc(c, f_);
  }
  void str(const std::string& s) {
    std::fputc('"', f_);
    for (char c : s) {
      if (c == '"' || c == '\\') {
        std::fputc('\\', f_);
        std::fputc(c, f_);
      } else if (static_cast<unsigned char>(c) < 0x20) {
        std::fprintf(f_, "\\u%04x",
                     static_cast<unsigned>(static_cast<unsigned char>(c)));
      } else {
        std::fputc(c, f_);
      }
    }
    std::fputc('"', f_);
  }
  FILE* f_;
  std::vector<bool> first_;
};

// ---------------------------------------------------------------- device
// description Recorded in every raw output so results can be matched to the
// NVML trace by PCI bus id (CUDA and NVML may enumerate GPUs in different
// orders).
inline void write_device(Json& j, int dev) {
  cudaDeviceProp p{};
  BENCH_CUDA(cudaGetDeviceProperties(&p, dev));
  char bus[32] = {0};
  BENCH_CUDA(cudaDeviceGetPCIBusId(bus, sizeof(bus), dev));
  int drv = 0, rt = 0;
  BENCH_CUDA(cudaDriverGetVersion(&drv));
  BENCH_CUDA(cudaRuntimeGetVersion(&rt));
  j.begin_obj("device");
  j.kv("cuda_index", dev);
  j.kv("name", std::string(p.name));
  j.kv("pci_bus_id", std::string(bus));
  j.kv("cc", std::to_string(p.major) + "." + std::to_string(p.minor));
  j.kv("sm_count", p.multiProcessorCount);
  j.kv("l2_bytes", p.l2CacheSize);
  j.kv("smem_per_block_limit", static_cast<int64_t>(p.sharedMemPerBlockOptin));
  j.kv("smem_per_sm", static_cast<int64_t>(p.sharedMemPerMultiprocessor));
  j.kv("smem_reserved_per_block",
       static_cast<int64_t>(p.reservedSharedMemPerBlock));
  j.kv("regs_per_sm", p.regsPerMultiprocessor);
  j.kv("max_threads_per_sm", p.maxThreadsPerMultiProcessor);
  j.kv("memory_bus_width_bits", p.memoryBusWidth);
  j.kv("total_global_mem", static_cast<int64_t>(p.totalGlobalMem));
  j.kv("driver_version", drv);
  j.kv("runtime_version", rt);
  j.end_obj();
}

// Launch a spin kernel before timing events so that, by the time the GPU
// reaches the start event, the measured work is already queued: event pairs
// then time GPU execution, not host submission latency.
__global__ void spin_ns(uint64_t ns) {
  uint64_t t0 = 0, t = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while (t - t0 < ns);
}

// 200 us: host-side event record + cudaGraphLaunch + event record take tens of
// microseconds (the typical driver cost, as seen in the graph-overhead mode of
// the hardware-ceilings runs), so this leaves an order of magnitude of margin.
constexpr uint64_t kSpinBeforeTimingNs = 200000;

}  // namespace bench
