// Micro-benchmarks: NEON vs portable kernels on one 4096-row batch (the
// executor's batch size) and on a cache-exceeding 16M-row column.

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "vecq/kernels.h"

using namespace vecq::kernels;

namespace {

struct Data {
  std::vector<int64_t> i64;
  std::vector<double> f64;
  std::vector<uint32_t> u32;
  std::vector<uint8_t> mask, out;
  std::vector<uint32_t> sel;
  explicit Data(size_t n) : i64(n), f64(n), u32(n), mask(n), out(n), sel(n) {
    std::mt19937_64 rng(7);
    for (size_t i = 0; i < n; ++i) {
      i64[i] = static_cast<int64_t>(rng() % 1000);
      f64[i] = static_cast<double>(rng() % 100000) / 100.0;
      u32[i] = static_cast<uint32_t>(rng() % 260);
      mask[i] = rng() % 2 ? 0xFF : 0x00;
    }
  }
};

Data& data(size_t n) {
  static Data small(4096);
  static Data* large = nullptr;
  if (n == 4096) return small;
  if (large == nullptr) large = new Data(n);
  return *large;
}

void set_bytes(benchmark::State& st, size_t n, size_t elem) {
  st.SetItemsProcessed(static_cast<int64_t>(st.iterations() * n));
  st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * n * elem));
}

template <bool Simd>
void BM_cmp_i64(benchmark::State& st) {
  auto n = static_cast<size_t>(st.range(0));
  Data& d = data(n);
  for (auto _ : st) {
    if constexpr (Simd)
      neon::cmp_i64(d.i64.data(), n, CmpOp::Gt, 500, d.out.data());
    else
      scalar::cmp_i64(d.i64.data(), n, CmpOp::Gt, 500, d.out.data());
    benchmark::DoNotOptimize(d.out.data());
    benchmark::ClobberMemory();
  }
  set_bytes(st, n, 8);
}

template <bool Simd>
void BM_cmp_f64(benchmark::State& st) {
  auto n = static_cast<size_t>(st.range(0));
  Data& d = data(n);
  for (auto _ : st) {
    if constexpr (Simd)
      neon::cmp_f64(d.f64.data(), n, CmpOp::Lt, 250.0, d.out.data());
    else
      scalar::cmp_f64(d.f64.data(), n, CmpOp::Lt, 250.0, d.out.data());
    benchmark::DoNotOptimize(d.out.data());
    benchmark::ClobberMemory();
  }
  set_bytes(st, n, 8);
}

template <bool Simd>
void BM_cmp_u32(benchmark::State& st) {
  auto n = static_cast<size_t>(st.range(0));
  Data& d = data(n);
  for (auto _ : st) {
    if constexpr (Simd)
      neon::cmp_u32(d.u32.data(), n, CmpOp::Eq, 42, d.out.data());
    else
      scalar::cmp_u32(d.u32.data(), n, CmpOp::Eq, 42, d.out.data());
    benchmark::DoNotOptimize(d.out.data());
    benchmark::ClobberMemory();
  }
  set_bytes(st, n, 4);
}

template <bool Simd>
void BM_sum_f64(benchmark::State& st) {
  auto n = static_cast<size_t>(st.range(0));
  Data& d = data(n);
  for (auto _ : st) {
    double s = Simd ? neon::masked_sum_f64(d.f64.data(), d.mask.data(), n)
                    : scalar::masked_sum_f64(d.f64.data(), d.mask.data(), n);
    benchmark::DoNotOptimize(s);
  }
  set_bytes(st, n, 9);
}

template <bool Simd>
void BM_sum_i64(benchmark::State& st) {
  auto n = static_cast<size_t>(st.range(0));
  Data& d = data(n);
  for (auto _ : st) {
    int64_t s = Simd ? neon::masked_sum_i64(d.i64.data(), d.mask.data(), n)
                     : scalar::masked_sum_i64(d.i64.data(), d.mask.data(), n);
    benchmark::DoNotOptimize(s);
  }
  set_bytes(st, n, 9);
}

template <bool Simd>
void BM_max_f64(benchmark::State& st) {
  auto n = static_cast<size_t>(st.range(0));
  Data& d = data(n);
  for (auto _ : st) {
    double v = 0;
    bool any = Simd ? neon::masked_max_f64(d.f64.data(), d.mask.data(), n, &v)
                    : scalar::masked_max_f64(d.f64.data(), d.mask.data(), n, &v);
    benchmark::DoNotOptimize(any);
    benchmark::DoNotOptimize(v);
  }
  set_bytes(st, n, 9);
}

template <bool Simd>
void BM_count(benchmark::State& st) {
  auto n = static_cast<size_t>(st.range(0));
  Data& d = data(n);
  for (auto _ : st) {
    size_t c = Simd ? neon::mask_count(d.mask.data(), n) : scalar::mask_count(d.mask.data(), n);
    benchmark::DoNotOptimize(c);
  }
  set_bytes(st, n, 1);
}

template <bool Simd>
void BM_mask_to_sel(benchmark::State& st) {
  auto n = static_cast<size_t>(st.range(0));
  Data& d = data(n);
  for (auto _ : st) {
    size_t c =
        Simd ? neon::mask_to_sel(d.mask.data(), n, d.sel.data()) : scalar::mask_to_sel(d.mask.data(), n, d.sel.data());
    benchmark::DoNotOptimize(c);
    benchmark::ClobberMemory();
  }
  set_bytes(st, n, 1);
}

}  // namespace

#ifdef VECQ_HAVE_NEON
#define VECQ_PAIR(fn)                                                                                 \
  BENCHMARK(fn<false>)->Name(#fn "/scalar")->Arg(4096)->Arg(16 << 20)->Unit(benchmark::kMicrosecond); \
  BENCHMARK(fn<true>)->Name(#fn "/neon")->Arg(4096)->Arg(16 << 20)->Unit(benchmark::kMicrosecond)
VECQ_PAIR(BM_cmp_i64);
VECQ_PAIR(BM_cmp_f64);
VECQ_PAIR(BM_cmp_u32);
VECQ_PAIR(BM_sum_i64);
VECQ_PAIR(BM_sum_f64);
VECQ_PAIR(BM_max_f64);
VECQ_PAIR(BM_count);
VECQ_PAIR(BM_mask_to_sel);
#endif

BENCHMARK_MAIN();
