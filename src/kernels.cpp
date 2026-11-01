#include "vecq/kernels.h"

#include <atomic>
#include <cstring>

namespace vecq::kernels {

namespace {
#ifdef VECQ_HAVE_NEON
std::atomic<bool> g_simd{true};
#else
std::atomic<bool> g_simd{false};
#endif
inline bool use_simd() { return g_simd.load(std::memory_order_relaxed); }
}  // namespace

bool simd_available() {
#ifdef VECQ_HAVE_NEON
  return true;
#else
  return false;
#endif
}

bool simd_enabled() { return use_simd(); }

void set_simd_enabled(bool on) { g_simd.store(on && simd_available(), std::memory_order_relaxed); }

#ifdef VECQ_HAVE_NEON
#define VECQ_DISPATCH(fn, ...) return use_simd() ? neon::fn(__VA_ARGS__) : scalar::fn(__VA_ARGS__)
#else
#define VECQ_DISPATCH(fn, ...) return scalar::fn(__VA_ARGS__)
#endif

void cmp_i64(const int64_t* x, size_t n, CmpOp op, int64_t c, uint8_t* out) {
  VECQ_DISPATCH(cmp_i64, x, n, op, c, out);
}
void cmp_f64(const double* x, size_t n, CmpOp op, double c, uint8_t* out) { VECQ_DISPATCH(cmp_f64, x, n, op, c, out); }
void cmp_u32(const uint32_t* x, size_t n, CmpOp op, uint32_t c, uint8_t* out) {
  VECQ_DISPATCH(cmp_u32, x, n, op, c, out);
}
void mask_and(uint8_t* dst, const uint8_t* src, size_t n) { VECQ_DISPATCH(mask_and, dst, src, n); }
void mask_or(uint8_t* dst, const uint8_t* src, size_t n) { VECQ_DISPATCH(mask_or, dst, src, n); }
size_t mask_count(const uint8_t* m, size_t n) { VECQ_DISPATCH(mask_count, m, n); }
int64_t masked_sum_i64(const int64_t* x, const uint8_t* m, size_t n) { VECQ_DISPATCH(masked_sum_i64, x, m, n); }
double masked_sum_f64(const double* x, const uint8_t* m, size_t n) { VECQ_DISPATCH(masked_sum_f64, x, m, n); }
bool masked_min_i64(const int64_t* x, const uint8_t* m, size_t n, int64_t* out) {
  VECQ_DISPATCH(masked_min_i64, x, m, n, out);
}
bool masked_max_i64(const int64_t* x, const uint8_t* m, size_t n, int64_t* out) {
  VECQ_DISPATCH(masked_max_i64, x, m, n, out);
}
bool masked_min_f64(const double* x, const uint8_t* m, size_t n, double* out) {
  VECQ_DISPATCH(masked_min_f64, x, m, n, out);
}
bool masked_max_f64(const double* x, const uint8_t* m, size_t n, double* out) {
  VECQ_DISPATCH(masked_max_f64, x, m, n, out);
}
size_t mask_to_sel(const uint8_t* m, size_t n, uint32_t* sel) { VECQ_DISPATCH(mask_to_sel, m, n, sel); }
void gather_u8(const uint32_t* codes, size_t n, const uint8_t* lut, uint8_t* out) {
  VECQ_DISPATCH(gather_u8, codes, n, lut, out);
}

#undef VECQ_DISPATCH

void mask_fill(uint8_t* m, size_t n, bool value) { std::memset(m, value ? 0xFF : 0x00, n); }

void mask_not(uint8_t* m, size_t n) {
  for (size_t i = 0; i < n; ++i) m[i] = static_cast<uint8_t>(~m[i]);
}

}  // namespace vecq::kernels
