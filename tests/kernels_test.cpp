#include "vecq/kernels.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

using namespace vecq::kernels;

namespace {

constexpr CmpOp kOps[] = {CmpOp::Eq, CmpOp::Ne, CmpOp::Lt, CmpOp::Le, CmpOp::Gt, CmpOp::Ge};

std::vector<size_t> sizes() { return {0, 1, 7, 8, 15, 16, 17, 31, 33, 100, 255, 4096, 4099, 16 * 255 * 2 + 5}; }

std::vector<uint8_t> random_mask(std::mt19937_64& rng, size_t n, int density_pct) {
  std::vector<uint8_t> m(n);
  for (auto& b : m) b = static_cast<int>(rng() % 100) < density_pct ? 0xFF : 0x00;
  return m;
}

}  // namespace

TEST(Kernels, ScalarCompareSemantics) {
  std::vector<int64_t> x = {-3, 0, 5, 5, 9};
  std::vector<uint8_t> out(x.size());
  scalar::cmp_i64(x.data(), x.size(), CmpOp::Ge, 5, out.data());
  EXPECT_EQ(out, (std::vector<uint8_t>{0, 0, 0xFF, 0xFF, 0xFF}));
  scalar::cmp_i64(x.data(), x.size(), CmpOp::Ne, 5, out.data());
  EXPECT_EQ(out, (std::vector<uint8_t>{0xFF, 0xFF, 0, 0, 0xFF}));
}

TEST(Kernels, MaskToSel) {
  std::vector<uint8_t> m = {0, 0xFF, 0, 0xFF, 0xFF};
  std::vector<uint32_t> sel(m.size());
  size_t k = mask_to_sel(m.data(), m.size(), sel.data());
  ASSERT_EQ(k, 3u);
  EXPECT_EQ(sel[0], 1u);
  EXPECT_EQ(sel[1], 3u);
  EXPECT_EQ(sel[2], 4u);
}

TEST(Kernels, MinMaxEmptySelection) {
  std::vector<int64_t> x(40, 7);
  std::vector<uint8_t> m(40, 0);
  int64_t out = 123;
  EXPECT_FALSE(masked_min_i64(x.data(), m.data(), x.size(), &out));
  EXPECT_EQ(out, 123);
  m[39] = 0xFF;
  EXPECT_TRUE(masked_max_i64(x.data(), m.data(), x.size(), &out));
  EXPECT_EQ(out, 7);
}

#ifdef VECQ_HAVE_NEON

TEST(NeonVsScalar, Compare) {
  std::mt19937_64 rng(1);
  for (size_t n : sizes()) {
    std::vector<int64_t> xi(n);
    std::vector<double> xf(n);
    std::vector<uint32_t> xu(n);
    for (size_t i = 0; i < n; ++i) {
      xi[i] = static_cast<int64_t>(rng() % 21) - 10;
      xf[i] = static_cast<double>(xi[i]) * 0.5;
      xu[i] = static_cast<uint32_t>(rng() % 7);
    }
    if (n > 3) {
      xi[0] = std::numeric_limits<int64_t>::min();
      xi[1] = std::numeric_limits<int64_t>::max();
      xf[2] = std::nan("");
      xu[3] = 0xFFFFFFFFu;
    }
    std::vector<uint8_t> a(n), b(n);
    for (CmpOp op : kOps) {
      scalar::cmp_i64(xi.data(), n, op, 3, a.data());
      neon::cmp_i64(xi.data(), n, op, 3, b.data());
      ASSERT_EQ(a, b) << "i64 n=" << n << " op=" << static_cast<int>(op);
      scalar::cmp_f64(xf.data(), n, op, 1.5, a.data());
      neon::cmp_f64(xf.data(), n, op, 1.5, b.data());
      ASSERT_EQ(a, b) << "f64 n=" << n << " op=" << static_cast<int>(op);
      scalar::cmp_u32(xu.data(), n, op, 3, a.data());
      neon::cmp_u32(xu.data(), n, op, 3, b.data());
      ASSERT_EQ(a, b) << "u32 n=" << n << " op=" << static_cast<int>(op);
    }
  }
}

TEST(NeonVsScalar, MaskOpsAndCount) {
  std::mt19937_64 rng(2);
  for (size_t n : sizes()) {
    for (int density : {0, 3, 50, 100}) {
      auto m1 = random_mask(rng, n, density);
      auto m2 = random_mask(rng, n, 50);
      ASSERT_EQ(scalar::mask_count(m1.data(), n), neon::mask_count(m1.data(), n)) << n;
      auto a = m1, b = m1;
      scalar::mask_and(a.data(), m2.data(), n);
      neon::mask_and(b.data(), m2.data(), n);
      ASSERT_EQ(a, b);
      a = m1, b = m1;
      scalar::mask_or(a.data(), m2.data(), n);
      neon::mask_or(b.data(), m2.data(), n);
      ASSERT_EQ(a, b);
      std::vector<uint32_t> s1(n + 1), s2(n + 1);
      size_t k1 = scalar::mask_to_sel(m1.data(), n, s1.data());
      size_t k2 = neon::mask_to_sel(m1.data(), n, s2.data());
      ASSERT_EQ(k1, k2);
      s1.resize(k1);
      s2.resize(k2);
      ASSERT_EQ(s1, s2);
    }
  }
}

TEST(NeonVsScalar, Aggregates) {
  std::mt19937_64 rng(3);
  for (size_t n : sizes()) {
    for (int density : {0, 1, 50, 100}) {
      std::vector<int64_t> xi(n);
      std::vector<double> xf(n);
      for (size_t i = 0; i < n; ++i) {
        xi[i] = static_cast<int64_t>(rng() % 2000001) - 1000000;
        xf[i] = static_cast<double>(static_cast<int64_t>(rng() % 200001) - 100000) / 7.0;
      }
      auto m = random_mask(rng, n, density);
      ASSERT_EQ(scalar::masked_sum_i64(xi.data(), m.data(), n), neon::masked_sum_i64(xi.data(), m.data(), n));
      double fs = scalar::masked_sum_f64(xf.data(), m.data(), n);
      double fn = neon::masked_sum_f64(xf.data(), m.data(), n);
      ASSERT_NEAR(fs, fn, 1e-9 * std::max(1.0, std::fabs(fs)));

      int64_t si = 0, ni = 0;
      double sf = 0, nf = 0;
      ASSERT_EQ(scalar::masked_min_i64(xi.data(), m.data(), n, &si), neon::masked_min_i64(xi.data(), m.data(), n, &ni));
      ASSERT_EQ(si, ni);
      ASSERT_EQ(scalar::masked_max_i64(xi.data(), m.data(), n, &si), neon::masked_max_i64(xi.data(), m.data(), n, &ni));
      ASSERT_EQ(si, ni);
      ASSERT_EQ(scalar::masked_min_f64(xf.data(), m.data(), n, &sf), neon::masked_min_f64(xf.data(), m.data(), n, &nf));
      ASSERT_EQ(sf, nf);
      ASSERT_EQ(scalar::masked_max_f64(xf.data(), m.data(), n, &sf), neon::masked_max_f64(xf.data(), m.data(), n, &nf));
      ASSERT_EQ(sf, nf);
    }
  }
}

TEST(NeonVsScalar, MinMaxIgnoresNaN) {
  std::vector<double> x(32, std::nan(""));
  x[20] = -4.0;
  x[3] = 9.0;
  std::vector<uint8_t> m(32, 0xFF);
  double s = 0, v = 0;
  ASSERT_TRUE(scalar::masked_min_f64(x.data(), m.data(), x.size(), &s));
  ASSERT_TRUE(neon::masked_min_f64(x.data(), m.data(), x.size(), &v));
  EXPECT_EQ(s, -4.0);
  EXPECT_EQ(v, -4.0);
  ASSERT_TRUE(neon::masked_max_f64(x.data(), m.data(), x.size(), &v));
  EXPECT_EQ(v, 9.0);
}

TEST(NeonVsScalar, DispatchToggle) {
  ASSERT_TRUE(simd_available());
  set_simd_enabled(false);
  EXPECT_FALSE(simd_enabled());
  set_simd_enabled(true);
  EXPECT_TRUE(simd_enabled());
}

#endif  // VECQ_HAVE_NEON
