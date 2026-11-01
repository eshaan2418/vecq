// Portable reference kernels. Written branch-free so the compiler is free to
// auto-vectorize them; the NEON versions must beat *this*, not a strawman.

#include <cstring>
#include <limits>

#include "vecq/kernels.h"

namespace vecq::kernels::scalar {

namespace {

inline uint8_t sel(bool b) { return static_cast<uint8_t>(-static_cast<int>(b)); }

// Branch-free select on the bit pattern: returns x where the mask byte is
// 0xFF and `other` otherwise. A plain `m ? x : other` compiles to a branch on
// clang/AArch64, which mispredicts on random masks.
inline double select_bits(uint8_t m, double x, double other) {
  uint64_t xb, ob;
  std::memcpy(&xb, &x, sizeof xb);
  std::memcpy(&ob, &other, sizeof ob);
  const uint64_t k = static_cast<uint64_t>(-static_cast<int64_t>(m & 1u));
  const uint64_t r = (xb & k) | (ob & ~k);
  double out;
  std::memcpy(&out, &r, sizeof out);
  return out;
}

template <class T, class F>
void cmp_loop(const T* x, size_t n, T c, uint8_t* out, F f) {
  for (size_t i = 0; i < n; ++i) out[i] = sel(f(x[i], c));
}

template <class T>
void cmp_dispatch(const T* x, size_t n, CmpOp op, T c, uint8_t* out) {
  switch (op) {
    case CmpOp::Eq: cmp_loop(x, n, c, out, [](T a, T b) { return a == b; }); break;
    case CmpOp::Ne: cmp_loop(x, n, c, out, [](T a, T b) { return a != b; }); break;
    case CmpOp::Lt: cmp_loop(x, n, c, out, [](T a, T b) { return a < b; }); break;
    case CmpOp::Le: cmp_loop(x, n, c, out, [](T a, T b) { return a <= b; }); break;
    case CmpOp::Gt: cmp_loop(x, n, c, out, [](T a, T b) { return a > b; }); break;
    case CmpOp::Ge: cmp_loop(x, n, c, out, [](T a, T b) { return a >= b; }); break;
  }
}

}  // namespace

void cmp_i64(const int64_t* x, size_t n, CmpOp op, int64_t c, uint8_t* out) { cmp_dispatch(x, n, op, c, out); }
void cmp_f64(const double* x, size_t n, CmpOp op, double c, uint8_t* out) { cmp_dispatch(x, n, op, c, out); }
void cmp_u32(const uint32_t* x, size_t n, CmpOp op, uint32_t c, uint8_t* out) { cmp_dispatch(x, n, op, c, out); }

void mask_and(uint8_t* dst, const uint8_t* src, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] &= src[i];
}
void mask_or(uint8_t* dst, const uint8_t* src, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] |= src[i];
}

size_t mask_count(const uint8_t* m, size_t n) {
  size_t c = 0;
  for (size_t i = 0; i < n; ++i) c += m[i] & 1u;
  return c;
}

int64_t masked_sum_i64(const int64_t* x, const uint8_t* m, size_t n) {
  // Unsigned accumulation: wraps like the NEON version instead of being UB.
  uint64_t s = 0;
  for (size_t i = 0; i < n; ++i) {
    s += static_cast<uint64_t>(x[i]) & static_cast<uint64_t>(-static_cast<int64_t>(m[i] & 1u));
  }
  return static_cast<int64_t>(s);
}

double masked_sum_f64(const double* x, const uint8_t* m, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; ++i) s += select_bits(m[i], x[i], 0.0);
  return s;
}

bool masked_min_i64(const int64_t* x, const uint8_t* m, size_t n, int64_t* out) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  int64_t best = kMax;
  uint8_t any = 0;
  for (size_t i = 0; i < n; ++i) {
    any |= m[i];
    int64_t c = m[i] ? x[i] : kMax;
    best = c < best ? c : best;
  }
  if (any) *out = best;
  return any != 0;
}

bool masked_max_i64(const int64_t* x, const uint8_t* m, size_t n, int64_t* out) {
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  int64_t best = kMin;
  uint8_t any = 0;
  for (size_t i = 0; i < n; ++i) {
    any |= m[i];
    int64_t c = m[i] ? x[i] : kMin;
    best = c > best ? c : best;
  }
  if (any) *out = best;
  return any != 0;
}

// NaN values are ignored (IEEE minNum/maxNum semantics), matching vminnmq.
bool masked_min_f64(const double* x, const uint8_t* m, size_t n, double* out) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  double best = kInf;
  uint8_t any = 0;
  for (size_t i = 0; i < n; ++i) {
    any |= m[i];
    double c = select_bits(m[i], x[i], kInf);
    best = c < best ? c : best;
  }
  if (any) *out = best;
  return any != 0;
}

bool masked_max_f64(const double* x, const uint8_t* m, size_t n, double* out) {
  constexpr double kNegInf = -std::numeric_limits<double>::infinity();
  double best = kNegInf;
  uint8_t any = 0;
  for (size_t i = 0; i < n; ++i) {
    any |= m[i];
    double c = select_bits(m[i], x[i], kNegInf);
    best = c > best ? c : best;
  }
  if (any) *out = best;
  return any != 0;
}

size_t mask_to_sel(const uint8_t* m, size_t n, uint32_t* sel_out) {
  size_t k = 0;
  for (size_t i = 0; i < n; ++i) {
    sel_out[k] = static_cast<uint32_t>(i);
    k += m[i] & 1u;
  }
  return k;
}

void gather_u8(const uint32_t* codes, size_t n, const uint8_t* lut, uint8_t* out) {
  for (size_t i = 0; i < n; ++i) out[i] = lut[codes[i]];
}

}  // namespace vecq::kernels::scalar
