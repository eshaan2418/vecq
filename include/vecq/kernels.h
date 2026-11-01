#pragma once

// Vectorized primitives over contiguous column slices.
//
// Masks are byte-per-row: 0xFF selects a row, 0x00 rejects it. Byte masks
// keep the NEON kernels simple (a 128-bit register holds 16 row decisions)
// and make AND/OR/count trivially vectorizable.
//
// Every kernel has a portable implementation in `scalar::` and, on AArch64,
// a hand-written NEON implementation in `neon::`. The unqualified functions
// dispatch on a process-wide switch so the same binary can benchmark both.

#include <cstddef>
#include <cstdint>

namespace vecq::kernels {

enum class CmpOp : uint8_t { Eq, Ne, Lt, Le, Gt, Ge };

bool simd_available();
bool simd_enabled();
void set_simd_enabled(bool on);

#define VECQ_KERNEL_DECLS                                                          \
  void cmp_i64(const int64_t* x, size_t n, CmpOp op, int64_t c, uint8_t* out);     \
  void cmp_f64(const double* x, size_t n, CmpOp op, double c, uint8_t* out);       \
  void cmp_u32(const uint32_t* x, size_t n, CmpOp op, uint32_t c, uint8_t* out);   \
  void mask_and(uint8_t* dst, const uint8_t* src, size_t n);                       \
  void mask_or(uint8_t* dst, const uint8_t* src, size_t n);                        \
  size_t mask_count(const uint8_t* m, size_t n);                                   \
  int64_t masked_sum_i64(const int64_t* x, const uint8_t* m, size_t n);            \
  double masked_sum_f64(const double* x, const uint8_t* m, size_t n);              \
  bool masked_min_i64(const int64_t* x, const uint8_t* m, size_t n, int64_t* out); \
  bool masked_max_i64(const int64_t* x, const uint8_t* m, size_t n, int64_t* out); \
  bool masked_min_f64(const double* x, const uint8_t* m, size_t n, double* out);   \
  bool masked_max_f64(const double* x, const uint8_t* m, size_t n, double* out);   \
  /* out[k] = index of k-th selected row; returns number selected. */              \
  size_t mask_to_sel(const uint8_t* m, size_t n, uint32_t* sel);                   \
  /* out[i] = lut[codes[i]] */                                                     \
  void gather_u8(const uint32_t* codes, size_t n, const uint8_t* lut, uint8_t* out);

VECQ_KERNEL_DECLS

namespace scalar {
VECQ_KERNEL_DECLS
}  // namespace scalar

#if defined(__ARM_NEON) || defined(__aarch64__)
#define VECQ_HAVE_NEON 1
namespace neon {
VECQ_KERNEL_DECLS
}  // namespace neon
#endif

#undef VECQ_KERNEL_DECLS

void mask_fill(uint8_t* m, size_t n, bool value);
void mask_not(uint8_t* m, size_t n);

}  // namespace vecq::kernels
