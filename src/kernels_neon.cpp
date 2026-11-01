// AArch64 NEON kernels.
//
// Comparison kernels evaluate 16 rows per iteration and narrow the 64-bit
// (or 32-bit) lane masks down to one byte per row with a tree of vmovn
// instructions. Aggregation kernels go the other way: 8 mask bytes are
// sign-extended to 64-bit lane masks (0xFF -> all ones) and ANDed with the
// values, so unselected rows contribute an additive identity without
// branches. Multiple accumulators hide the FP add latency.

#include "vecq/kernels.h"

#ifdef VECQ_HAVE_NEON

#include <arm_neon.h>

#include <limits>

namespace vecq::kernels::neon {

namespace {

inline uint8x16_t narrow64(uint64x2_t r0, uint64x2_t r1, uint64x2_t r2, uint64x2_t r3, uint64x2_t r4, uint64x2_t r5,
                           uint64x2_t r6, uint64x2_t r7) {
  uint32x4_t a = vcombine_u32(vmovn_u64(r0), vmovn_u64(r1));
  uint32x4_t b = vcombine_u32(vmovn_u64(r2), vmovn_u64(r3));
  uint32x4_t c = vcombine_u32(vmovn_u64(r4), vmovn_u64(r5));
  uint32x4_t d = vcombine_u32(vmovn_u64(r6), vmovn_u64(r7));
  uint16x8_t p = vcombine_u16(vmovn_u32(a), vmovn_u32(b));
  uint16x8_t q = vcombine_u16(vmovn_u32(c), vmovn_u32(d));
  return vcombine_u8(vmovn_u16(p), vmovn_u16(q));
}

// Sign-extend 8 mask bytes into four 2-lane 64-bit masks.
struct Mask8 {
  uint64x2_t m0, m1, m2, m3;
};
inline Mask8 widen8(const uint8_t* m) {
  int16x8_t w = vmovl_s8(vreinterpret_s8_u8(vld1_u8(m)));
  int32x4_t lo = vmovl_s16(vget_low_s16(w));
  int32x4_t hi = vmovl_s16(vget_high_s16(w));
  return {vreinterpretq_u64_s64(vmovl_s32(vget_low_s32(lo))), vreinterpretq_u64_s64(vmovl_s32(vget_high_s32(lo))),
          vreinterpretq_u64_s64(vmovl_s32(vget_low_s32(hi))), vreinterpretq_u64_s64(vmovl_s32(vget_high_s32(hi)))};
}

template <bool Invert, class Load, class Vec, class F>
inline void cmp64_loop(const typename Load::elem* x, size_t n, Vec cv, uint8_t* out, F f) {
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const auto* p = x + i;
    uint8x16_t r =
        narrow64(f(Load::ld(p + 0), cv), f(Load::ld(p + 2), cv), f(Load::ld(p + 4), cv), f(Load::ld(p + 6), cv),
                 f(Load::ld(p + 8), cv), f(Load::ld(p + 10), cv), f(Load::ld(p + 12), cv), f(Load::ld(p + 14), cv));
    if constexpr (Invert) r = vmvnq_u8(r);
    vst1q_u8(out + i, r);
  }
  // Tail handled by the caller with the scalar kernel.
  (void)n;
}

struct LoadI64 {
  using elem = int64_t;
  static int64x2_t ld(const int64_t* p) { return vld1q_s64(p); }
};
struct LoadF64 {
  using elem = double;
  static float64x2_t ld(const double* p) { return vld1q_f64(p); }
};

inline size_t floor16(size_t n) { return n & ~static_cast<size_t>(15); }

}  // namespace

void cmp_i64(const int64_t* x, size_t n, CmpOp op, int64_t c, uint8_t* out) {
  const int64x2_t cv = vdupq_n_s64(c);
  switch (op) {
    case CmpOp::Eq:
      cmp64_loop<false, LoadI64>(x, n, cv, out, [](int64x2_t a, int64x2_t b) { return vceqq_s64(a, b); });
      break;
    case CmpOp::Ne:
      cmp64_loop<true, LoadI64>(x, n, cv, out, [](int64x2_t a, int64x2_t b) { return vceqq_s64(a, b); });
      break;
    case CmpOp::Lt:
      cmp64_loop<false, LoadI64>(x, n, cv, out, [](int64x2_t a, int64x2_t b) { return vcltq_s64(a, b); });
      break;
    case CmpOp::Le:
      cmp64_loop<false, LoadI64>(x, n, cv, out, [](int64x2_t a, int64x2_t b) { return vcleq_s64(a, b); });
      break;
    case CmpOp::Gt:
      cmp64_loop<false, LoadI64>(x, n, cv, out, [](int64x2_t a, int64x2_t b) { return vcgtq_s64(a, b); });
      break;
    case CmpOp::Ge:
      cmp64_loop<false, LoadI64>(x, n, cv, out, [](int64x2_t a, int64x2_t b) { return vcgeq_s64(a, b); });
      break;
  }
  const size_t done = floor16(n);
  scalar::cmp_i64(x + done, n - done, op, c, out + done);
}

void cmp_f64(const double* x, size_t n, CmpOp op, double c, uint8_t* out) {
  const float64x2_t cv = vdupq_n_f64(c);
  switch (op) {
    case CmpOp::Eq:
      cmp64_loop<false, LoadF64>(x, n, cv, out, [](float64x2_t a, float64x2_t b) { return vceqq_f64(a, b); });
      break;
    case CmpOp::Ne:
      cmp64_loop<true, LoadF64>(x, n, cv, out, [](float64x2_t a, float64x2_t b) { return vceqq_f64(a, b); });
      break;
    case CmpOp::Lt:
      cmp64_loop<false, LoadF64>(x, n, cv, out, [](float64x2_t a, float64x2_t b) { return vcltq_f64(a, b); });
      break;
    case CmpOp::Le:
      cmp64_loop<false, LoadF64>(x, n, cv, out, [](float64x2_t a, float64x2_t b) { return vcleq_f64(a, b); });
      break;
    case CmpOp::Gt:
      cmp64_loop<false, LoadF64>(x, n, cv, out, [](float64x2_t a, float64x2_t b) { return vcgtq_f64(a, b); });
      break;
    case CmpOp::Ge:
      cmp64_loop<false, LoadF64>(x, n, cv, out, [](float64x2_t a, float64x2_t b) { return vcgeq_f64(a, b); });
      break;
  }
  const size_t done = floor16(n);
  scalar::cmp_f64(x + done, n - done, op, c, out + done);
}

void cmp_u32(const uint32_t* x, size_t n, CmpOp op, uint32_t c, uint8_t* out) {
  const uint32x4_t cv = vdupq_n_u32(c);
  auto run = [&](auto f, bool invert) {
    for (size_t i = 0; i + 16 <= n; i += 16) {
      uint16x8_t p = vcombine_u16(vmovn_u32(f(vld1q_u32(x + i + 0), cv)), vmovn_u32(f(vld1q_u32(x + i + 4), cv)));
      uint16x8_t q = vcombine_u16(vmovn_u32(f(vld1q_u32(x + i + 8), cv)), vmovn_u32(f(vld1q_u32(x + i + 12), cv)));
      uint8x16_t r = vcombine_u8(vmovn_u16(p), vmovn_u16(q));
      if (invert) r = vmvnq_u8(r);
      vst1q_u8(out + i, r);
    }
  };
  switch (op) {
    case CmpOp::Eq: run([](uint32x4_t a, uint32x4_t b) { return vceqq_u32(a, b); }, false); break;
    case CmpOp::Ne: run([](uint32x4_t a, uint32x4_t b) { return vceqq_u32(a, b); }, true); break;
    case CmpOp::Lt: run([](uint32x4_t a, uint32x4_t b) { return vcltq_u32(a, b); }, false); break;
    case CmpOp::Le: run([](uint32x4_t a, uint32x4_t b) { return vcleq_u32(a, b); }, false); break;
    case CmpOp::Gt: run([](uint32x4_t a, uint32x4_t b) { return vcgtq_u32(a, b); }, false); break;
    case CmpOp::Ge: run([](uint32x4_t a, uint32x4_t b) { return vcgeq_u32(a, b); }, false); break;
  }
  const size_t done = floor16(n);
  scalar::cmp_u32(x + done, n - done, op, c, out + done);
}

void mask_and(uint8_t* dst, const uint8_t* src, size_t n) {
  size_t i = 0;
  for (; i + 16 <= n; i += 16) vst1q_u8(dst + i, vandq_u8(vld1q_u8(dst + i), vld1q_u8(src + i)));
  scalar::mask_and(dst + i, src + i, n - i);
}

void mask_or(uint8_t* dst, const uint8_t* src, size_t n) {
  size_t i = 0;
  for (; i + 16 <= n; i += 16) vst1q_u8(dst + i, vorrq_u8(vld1q_u8(dst + i), vld1q_u8(src + i)));
  scalar::mask_or(dst + i, src + i, n - i);
}

size_t mask_count(const uint8_t* m, size_t n) {
  size_t total = 0;
  size_t i = 0;
  while (i + 16 <= n) {
    // Each lane accumulates at most 255 ones before it must be flushed.
    uint8x16_t acc = vdupq_n_u8(0);
    size_t blocks = 0;
    for (; i + 16 <= n && blocks < 255; i += 16, ++blocks) {
      acc = vaddq_u8(acc, vshrq_n_u8(vld1q_u8(m + i), 7));
    }
    total += vaddlvq_u8(acc);
  }
  return total + scalar::mask_count(m + i, n - i);
}

int64_t masked_sum_i64(const int64_t* x, const uint8_t* m, size_t n) {
  int64x2_t a0 = vdupq_n_s64(0), a1 = a0, a2 = a0, a3 = a0;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    Mask8 k = widen8(m + i);
    a0 = vaddq_s64(a0, vandq_s64(vld1q_s64(x + i + 0), vreinterpretq_s64_u64(k.m0)));
    a1 = vaddq_s64(a1, vandq_s64(vld1q_s64(x + i + 2), vreinterpretq_s64_u64(k.m1)));
    a2 = vaddq_s64(a2, vandq_s64(vld1q_s64(x + i + 4), vreinterpretq_s64_u64(k.m2)));
    a3 = vaddq_s64(a3, vandq_s64(vld1q_s64(x + i + 6), vreinterpretq_s64_u64(k.m3)));
  }
  int64x2_t s = vaddq_s64(vaddq_s64(a0, a1), vaddq_s64(a2, a3));
  auto total = static_cast<uint64_t>(vaddvq_s64(s));
  total += static_cast<uint64_t>(scalar::masked_sum_i64(x + i, m + i, n - i));
  return static_cast<int64_t>(total);
}

double masked_sum_f64(const double* x, const uint8_t* m, size_t n) {
  float64x2_t a0 = vdupq_n_f64(0.0), a1 = a0, a2 = a0, a3 = a0;
  size_t i = 0;
  auto masked = [](const double* p, uint64x2_t k) {
    return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(vld1q_f64(p)), k));
  };
  for (; i + 8 <= n; i += 8) {
    Mask8 k = widen8(m + i);
    a0 = vaddq_f64(a0, masked(x + i + 0, k.m0));
    a1 = vaddq_f64(a1, masked(x + i + 2, k.m1));
    a2 = vaddq_f64(a2, masked(x + i + 4, k.m2));
    a3 = vaddq_f64(a3, masked(x + i + 6, k.m3));
  }
  float64x2_t s = vaddq_f64(vaddq_f64(a0, a1), vaddq_f64(a2, a3));
  return vaddvq_f64(s) + scalar::masked_sum_f64(x + i, m + i, n - i);
}

namespace {

template <bool IsMin>
bool minmax_i64(const int64_t* x, const uint8_t* m, size_t n, int64_t* out) {
  constexpr int64_t kInit = IsMin ? std::numeric_limits<int64_t>::max() : std::numeric_limits<int64_t>::min();
  const int64x2_t init = vdupq_n_s64(kInit);
  // Four independent accumulators break the loop-carried compare/select chain.
  int64x2_t acc[4] = {init, init, init, init};
  uint8x8_t any8 = vdup_n_u8(0);
  size_t i = 0;
  auto step = [&](int64x2_t& a, const int64_t* p, uint64x2_t k) {
    int64x2_t cand = vbslq_s64(k, vld1q_s64(p), init);
    uint64x2_t better = IsMin ? vcltq_s64(cand, a) : vcgtq_s64(cand, a);
    a = vbslq_s64(better, cand, a);
  };
  for (; i + 8 <= n; i += 8) {
    any8 = vorr_u8(any8, vld1_u8(m + i));
    Mask8 k = widen8(m + i);
    step(acc[0], x + i + 0, k.m0);
    step(acc[1], x + i + 2, k.m1);
    step(acc[2], x + i + 4, k.m2);
    step(acc[3], x + i + 6, k.m3);
  }
  for (int j = 1; j < 4; ++j) {
    uint64x2_t better = IsMin ? vcltq_s64(acc[j], acc[0]) : vcgtq_s64(acc[j], acc[0]);
    acc[0] = vbslq_s64(better, acc[j], acc[0]);
  }
  const int64x2_t& merged = acc[0];
  bool found = vmaxv_u8(any8) != 0;
  int64_t best = kInit;
  if (found) {
    int64_t l0 = vgetq_lane_s64(merged, 0), l1 = vgetq_lane_s64(merged, 1);
    best = IsMin ? (l0 < l1 ? l0 : l1) : (l0 > l1 ? l0 : l1);
  }
  int64_t tail = 0;
  bool tail_found =
      IsMin ? scalar::masked_min_i64(x + i, m + i, n - i, &tail) : scalar::masked_max_i64(x + i, m + i, n - i, &tail);
  if (tail_found) {
    best = !found ? tail : (IsMin ? (tail < best ? tail : best) : (tail > best ? tail : best));
    found = true;
  }
  if (found) *out = best;
  return found;
}

template <bool IsMin>
bool minmax_f64(const double* x, const uint8_t* m, size_t n, double* out) {
  constexpr double kInit = IsMin ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
  const float64x2_t init = vdupq_n_f64(kInit);
  float64x2_t acc[4] = {init, init, init, init};
  uint8x8_t any8 = vdup_n_u8(0);
  size_t i = 0;
  // minnm/maxnm return the numeric operand when the other is NaN.
  auto step = [&](float64x2_t& a, const double* p, uint64x2_t k) {
    float64x2_t cand = vbslq_f64(k, vld1q_f64(p), init);
    a = IsMin ? vminnmq_f64(a, cand) : vmaxnmq_f64(a, cand);
  };
  for (; i + 8 <= n; i += 8) {
    any8 = vorr_u8(any8, vld1_u8(m + i));
    Mask8 k = widen8(m + i);
    step(acc[0], x + i + 0, k.m0);
    step(acc[1], x + i + 2, k.m1);
    step(acc[2], x + i + 4, k.m2);
    step(acc[3], x + i + 6, k.m3);
  }
  for (int j = 1; j < 4; ++j) acc[0] = IsMin ? vminnmq_f64(acc[0], acc[j]) : vmaxnmq_f64(acc[0], acc[j]);
  bool found = vmaxv_u8(any8) != 0;
  double best = IsMin ? vminnmvq_f64(acc[0]) : vmaxnmvq_f64(acc[0]);
  double tail = 0;
  bool tail_found =
      IsMin ? scalar::masked_min_f64(x + i, m + i, n - i, &tail) : scalar::masked_max_f64(x + i, m + i, n - i, &tail);
  if (tail_found) {
    if (!found) {
      best = tail;
    } else if (IsMin ? tail < best : tail > best) {
      best = tail;
    }
    found = true;
  }
  if (found) *out = best;
  return found;
}

}  // namespace

bool masked_min_i64(const int64_t* x, const uint8_t* m, size_t n, int64_t* out) {
  return minmax_i64<true>(x, m, n, out);
}
bool masked_max_i64(const int64_t* x, const uint8_t* m, size_t n, int64_t* out) {
  return minmax_i64<false>(x, m, n, out);
}
bool masked_min_f64(const double* x, const uint8_t* m, size_t n, double* out) { return minmax_f64<true>(x, m, n, out); }
bool masked_max_f64(const double* x, const uint8_t* m, size_t n, double* out) {
  return minmax_f64<false>(x, m, n, out);
}

size_t mask_to_sel(const uint8_t* m, size_t n, uint32_t* sel) {
  size_t k = 0;
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    uint8x16_t v = vld1q_u8(m + i);
    if (vmaxvq_u8(v) == 0) continue;  // nothing selected: skip 16 rows at once
    if (vminvq_u8(v) == 0xFF) {       // everything selected: emit a run
      const auto base = static_cast<uint32_t>(i);
      for (uint32_t j = 0; j < 16; ++j) sel[k + j] = base + j;
      k += 16;
      continue;
    }
    for (size_t j = i; j < i + 16; ++j) {
      sel[k] = static_cast<uint32_t>(j);
      k += m[j] & 1u;
    }
  }
  for (; i < n; ++i) {
    sel[k] = static_cast<uint32_t>(i);
    k += m[i] & 1u;
  }
  return k;
}

void gather_u8(const uint32_t* codes, size_t n, const uint8_t* lut, uint8_t* out) {
  // No byte gather in NEON; the scalar loop is already load-bound.
  scalar::gather_u8(codes, n, lut, out);
}

}  // namespace vecq::kernels::neon

#endif  // VECQ_HAVE_NEON
