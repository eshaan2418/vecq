#include "vecq/exec.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

#include "vecq/kernels.h"
#include "vecq/parallel.h"

namespace vecq {

using kernels::CmpOp;
using sql::AggFunc;
using sql::Literal;
using sql::Pred;
using sql::SelectItem;

namespace {

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error(m); }

const Column& column_of(const Table& t, const std::string& name) {
  int i = t.find(name);
  if (i < 0) fail("unknown column '" + name + "'");
  return t.columns[static_cast<size_t>(i)];
}

// ---------------------------------------------------------------------------
// LIKE with % and _ (case-sensitive). Greedy with backtracking to the last %.

bool like_match(std::string_view s, std::string_view p) {
  size_t si = 0, pi = 0, star = std::string_view::npos, mark = 0;
  while (si < s.size()) {
    if (pi < p.size() && (p[pi] == '_' || p[pi] == s[si])) {
      ++si;
      ++pi;
    } else if (pi < p.size() && p[pi] == '%') {
      star = pi++;
      mark = si;
    } else if (star != std::string_view::npos) {
      pi = star + 1;
      si = ++mark;
    } else {
      return false;
    }
  }
  while (pi < p.size() && p[pi] == '%') ++pi;
  return pi == p.size();
}

template <class T>
bool apply_cmp(const T& a, CmpOp op, const T& b) {
  switch (op) {
    case CmpOp::Eq: return a == b;
    case CmpOp::Ne: return a != b;
    case CmpOp::Lt: return a < b;
    case CmpOp::Le: return a <= b;
    case CmpOp::Gt: return a > b;
    case CmpOp::Ge: return a >= b;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Bound predicates

struct BPred {
  enum class Kind : uint8_t { CmpI64, CmpF64, CmpU32, Lut, Const, IsNull, And, Or } kind = Kind::Const;
  const Column* col = nullptr;
  CmpOp op = CmpOp::Eq;
  int64_t ci = 0;
  double cf = 0;
  uint32_t cu = 0;
  std::vector<uint8_t> lut;
  bool value = false;    // Const
  bool negated = false;  // IS NOT NULL
  size_t id = 0;
  std::unique_ptr<BPred> l, r;
};

// Rewrites `int_col op float_literal` into an exact integer comparison or a
// constant, so the INT64 kernel can be used without converting the column.
void bind_int_vs_float(BPred& b, CmpOp op, double f) {
  constexpr double kMax = 9223372036854775807.0;  // 2^63
  if (std::isnan(f)) {
    b.kind = BPred::Kind::Const;
    b.value = op == CmpOp::Ne;
    return;
  }
  if (f >= kMax || f < -kMax) {
    const bool above = f > 0;  // literal is above every int64
    b.kind = BPred::Kind::Const;
    switch (op) {
      case CmpOp::Eq: b.value = false; break;
      case CmpOp::Ne: b.value = true; break;
      case CmpOp::Lt:
      case CmpOp::Le: b.value = above; break;
      case CmpOp::Gt:
      case CmpOp::Ge: b.value = !above; break;
    }
    return;
  }
  if (std::floor(f) == f) {
    b.kind = BPred::Kind::CmpI64;
    b.op = op;
    b.ci = static_cast<int64_t>(f);
    return;
  }
  b.kind = BPred::Kind::CmpI64;
  switch (op) {
    case CmpOp::Eq:
      b.kind = BPred::Kind::Const;
      b.value = false;
      break;
    case CmpOp::Ne:
      b.kind = BPred::Kind::Const;
      b.value = true;
      break;
    case CmpOp::Lt:
    case CmpOp::Le:
      b.op = CmpOp::Le;
      b.ci = static_cast<int64_t>(std::floor(f));
      break;
    case CmpOp::Gt:
    case CmpOp::Ge:
      b.op = CmpOp::Ge;
      b.ci = static_cast<int64_t>(std::ceil(f));
      break;
  }
}

std::unique_ptr<BPred> bind(const Table& t, const Pred& p, size_t& next_id) {
  auto b = std::make_unique<BPred>();
  b->id = next_id++;
  switch (p.kind) {
    case Pred::Kind::And:
    case Pred::Kind::Or:
      b->kind = p.kind == Pred::Kind::And ? BPred::Kind::And : BPred::Kind::Or;
      b->l = bind(t, *p.lhs, next_id);
      b->r = bind(t, *p.rhs, next_id);
      return b;
    case Pred::Kind::IsNull:
      b->kind = BPred::Kind::IsNull;
      b->col = &column_of(t, p.column);
      b->negated = p.negated;
      return b;
    case Pred::Kind::Like: {
      b->col = &column_of(t, p.column);
      if (b->col->type != Type::String) fail("LIKE requires a STRING column: " + p.column);
      b->kind = BPred::Kind::Lut;
      b->lut.resize(std::max<size_t>(1, b->col->dict.size()));
      for (size_t k = 0; k < b->col->dict.size(); ++k) {
        b->lut[k] = (like_match(b->col->dict[k], p.lit.s) != p.negated) ? 0xFF : 0x00;
      }
      return b;
    }
    case Pred::Kind::Cmp: break;
  }

  const Column& c = column_of(t, p.column);
  b->col = &c;
  const Literal& lit = p.lit;
  switch (c.type) {
    case Type::Int64:
      if (lit.kind == Literal::Kind::String) fail("cannot compare INT64 column '" + c.name + "' with a string");
      if (lit.kind == Literal::Kind::Int) {
        b->kind = BPred::Kind::CmpI64;
        b->op = p.op;
        b->ci = lit.i;
      } else {
        bind_int_vs_float(*b, p.op, lit.f);
      }
      break;
    case Type::Float64:
      if (lit.kind == Literal::Kind::String) fail("cannot compare FLOAT64 column '" + c.name + "' with a string");
      b->kind = BPred::Kind::CmpF64;
      b->op = p.op;
      b->cf = lit.kind == Literal::Kind::Int ? static_cast<double>(lit.i) : lit.f;
      break;
    case Type::String: {
      if (lit.kind != Literal::Kind::String) fail("cannot compare STRING column '" + c.name + "' with a number");
      if (p.op == CmpOp::Eq || p.op == CmpOp::Ne) {
        auto it = std::find(c.dict.begin(), c.dict.end(), lit.s);
        if (it == c.dict.end()) {
          b->kind = BPred::Kind::Const;
          b->value = p.op == CmpOp::Ne;
        } else {
          b->kind = BPred::Kind::CmpU32;
          b->op = p.op;
          b->cu = static_cast<uint32_t>(it - c.dict.begin());
        }
      } else {
        // Ordering on strings: evaluate once per dictionary entry.
        b->kind = BPred::Kind::Lut;
        b->lut.resize(std::max<size_t>(1, c.dict.size()));
        for (size_t k = 0; k < c.dict.size(); ++k) {
          b->lut[k] = apply_cmp(std::string_view(c.dict[k]), p.op, std::string_view(lit.s)) ? 0xFF : 0x00;
        }
      }
      break;
    }
  }
  return b;
}

// Scratch masks for one worker, one per predicate node.
struct Scratch {
  std::vector<std::vector<uint8_t>> node;
  std::vector<uint8_t> mask, tmp;
  std::vector<uint32_t> sel;
  std::vector<uint32_t> gids;
};

void eval(const BPred& p, size_t start, size_t n, uint8_t* out, Scratch& s) {
  using K = BPred::Kind;
  switch (p.kind) {
    case K::And:
    case K::Or: {
      eval(*p.l, start, n, out, s);
      uint8_t* rhs = s.node[p.r->id].data();
      eval(*p.r, start, n, rhs, s);
      if (p.kind == K::And) {
        kernels::mask_and(out, rhs, n);
      } else {
        kernels::mask_or(out, rhs, n);
      }
      return;
    }
    case K::IsNull:
      if (p.col->has_nulls()) {
        std::memcpy(out, p.col->valid.data() + start, n);
        if (!p.negated) kernels::mask_not(out, n);
      } else {
        kernels::mask_fill(out, n, p.negated);
      }
      return;
    case K::CmpI64: kernels::cmp_i64(p.col->i64.data() + start, n, p.op, p.ci, out); break;
    case K::CmpF64: kernels::cmp_f64(p.col->f64.data() + start, n, p.op, p.cf, out); break;
    case K::CmpU32: kernels::cmp_u32(p.col->codes.data() + start, n, p.op, p.cu, out); break;
    case K::Lut: kernels::gather_u8(p.col->codes.data() + start, n, p.lut.data(), out); break;
    case K::Const: kernels::mask_fill(out, n, p.value); break;
  }
  // Comparisons against NULL are never true.
  if (p.col != nullptr && p.col->has_nulls()) kernels::mask_and(out, p.col->valid.data() + start, n);
}

// ---------------------------------------------------------------------------
// Aggregation

struct AggSpec {
  AggFunc func;
  const Column* col;  // nullptr for COUNT(*)
};

struct Acc {
  int64_t count = 0;
  uint64_t isum = 0;  // wrapping
  double fsum = 0;
  int64_t imin = std::numeric_limits<int64_t>::max();
  int64_t imax = std::numeric_limits<int64_t>::min();
  double fmin = std::numeric_limits<double>::infinity();
  double fmax = -std::numeric_limits<double>::infinity();
  bool has = false;

  void merge(const Acc& o) {
    count += o.count;
    isum += o.isum;
    fsum += o.fsum;
    imin = std::min(imin, o.imin);
    imax = std::max(imax, o.imax);
    if (o.fmin < fmin) fmin = o.fmin;
    if (o.fmax > fmax) fmax = o.fmax;
    has |= o.has;
  }
};

Value finalize(const AggSpec& a, const Acc& acc) {
  const bool is_int = a.col != nullptr && a.col->type == Type::Int64;
  switch (a.func) {
    case AggFunc::Count: return acc.count;
    case AggFunc::Sum:
      if (acc.count == 0) return std::monostate{};
      if (is_int) return static_cast<int64_t>(acc.isum);
      return acc.fsum;
    case AggFunc::Avg:
      if (acc.count == 0) return std::monostate{};
      if (is_int) return static_cast<double>(static_cast<int64_t>(acc.isum)) / static_cast<double>(acc.count);
      return acc.fsum / static_cast<double>(acc.count);
    case AggFunc::Min:
      if (!acc.has) return std::monostate{};
      if (is_int) return acc.imin;
      return acc.fmin;
    case AggFunc::Max:
      if (!acc.has) return std::monostate{};
      if (is_int) return acc.imax;
      return acc.fmax;
  }
  return std::monostate{};
}

// Vectorized update of one accumulator from a masked batch.
void update_masked(const AggSpec& a, Acc& acc, size_t start, size_t n, const uint8_t* mask, Scratch& s) {
  if (a.col == nullptr) {
    acc.count += static_cast<int64_t>(kernels::mask_count(mask, n));
    return;
  }
  const uint8_t* m = mask;
  if (a.col->has_nulls()) {
    std::memcpy(s.tmp.data(), mask, n);
    kernels::mask_and(s.tmp.data(), a.col->valid.data() + start, n);
    m = s.tmp.data();
  }
  const bool is_int = a.col->type == Type::Int64;
  switch (a.func) {
    case AggFunc::Count: acc.count += static_cast<int64_t>(kernels::mask_count(m, n)); break;
    case AggFunc::Sum:
    case AggFunc::Avg:
      acc.count += static_cast<int64_t>(kernels::mask_count(m, n));
      if (is_int) {
        acc.isum += static_cast<uint64_t>(kernels::masked_sum_i64(a.col->i64.data() + start, m, n));
      } else {
        acc.fsum += kernels::masked_sum_f64(a.col->f64.data() + start, m, n);
      }
      break;
    case AggFunc::Min:
    case AggFunc::Max: {
      const bool is_min = a.func == AggFunc::Min;
      if (is_int) {
        int64_t v;
        bool any = is_min ? kernels::masked_min_i64(a.col->i64.data() + start, m, n, &v)
                          : kernels::masked_max_i64(a.col->i64.data() + start, m, n, &v);
        if (any) {
          acc.has = true;
          if (is_min)
            acc.imin = std::min(acc.imin, v);
          else
            acc.imax = std::max(acc.imax, v);
        }
      } else {
        double v;
        bool any = is_min ? kernels::masked_min_f64(a.col->f64.data() + start, m, n, &v)
                          : kernels::masked_max_f64(a.col->f64.data() + start, m, n, &v);
        if (any) {
          acc.has = true;
          if (is_min && v < acc.fmin) acc.fmin = v;
          if (!is_min && v > acc.fmax) acc.fmax = v;
        }
      }
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// Group-by hash table

constexpr size_t kMaxKeys = 4;

struct Key {
  int64_t v[kMaxKeys] = {0, 0, 0, 0};
  uint8_t nulls = 0;
  bool operator==(const Key& o) const {
    return nulls == o.nulls && v[0] == o.v[0] && v[1] == o.v[1] && v[2] == o.v[2] && v[3] == o.v[3];
  }
};

inline uint64_t mix(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

inline uint64_t hash_key(const Key& k) {
  uint64_t h = 0x9e3779b97f4a7c15ULL ^ k.nulls;
  for (int64_t v : k.v) h = mix(h ^ static_cast<uint64_t>(v));
  return h;
}

inline int64_t key_bits(const Column& c, size_t row) {
  switch (c.type) {
    case Type::Int64: return c.i64[row];
    case Type::String: return c.codes[row];
    case Type::Float64: {
      double d = c.f64[row];
      if (d == 0.0) d = 0.0;  // fold -0.0 into 0.0
      int64_t bits;
      std::memcpy(&bits, &d, sizeof bits);
      return bits;
    }
  }
  return 0;
}

class GroupTable {
 public:
  GroupTable(const std::vector<const Column*>& keys, size_t num_aggs) : keys_(keys), num_aggs_(num_aggs) {
    dense_ = keys.size() == 1 && keys[0]->type == Type::String;
    if (dense_) {
      dense_slots_.assign(keys[0]->dict.size() + 1, -1);  // last slot = NULL
    } else {
      slots_.assign(64, -1);
    }
  }

  [[nodiscard]] size_t size() const { return groups_.size(); }
  [[nodiscard]] const Key& key(size_t g) const { return groups_[g]; }
  Acc* accs(size_t g) { return &accs_[g * num_aggs_]; }

  Key make_key(size_t row) const {
    Key k;
    for (size_t i = 0; i < keys_.size(); ++i) {
      const Column& c = *keys_[i];
      if (!c.is_valid(row)) {
        k.nulls |= static_cast<uint8_t>(1u << i);
      } else {
        k.v[i] = key_bits(c, row);
      }
    }
    return k;
  }

  uint32_t find_or_insert(const Key& k) {
    if (dense_) {
      size_t slot = (k.nulls & 1u) ? dense_slots_.size() - 1 : static_cast<size_t>(k.v[0]);
      int32_t& g = dense_slots_[slot];
      if (g < 0) g = static_cast<int32_t>(insert(k));
      return static_cast<uint32_t>(g);
    }
    size_t mask = slots_.size() - 1;
    for (size_t i = hash_key(k) & mask;; i = (i + 1) & mask) {
      int32_t g = slots_[i];
      if (g < 0) {
        auto ng = static_cast<int32_t>(insert(k));
        slots_[i] = ng;
        if (groups_.size() * 2 > slots_.size()) grow();
        return static_cast<uint32_t>(ng);
      }
      if (groups_[static_cast<size_t>(g)] == k) return static_cast<uint32_t>(g);
    }
  }

  // Group ids for selected rows of one batch.
  void lookup_batch(size_t start, const uint32_t* sel, size_t nsel, uint32_t* gids) {
    if (dense_ && !keys_[0]->has_nulls()) {
      const uint32_t* codes = keys_[0]->codes.data() + start;
      for (size_t k = 0; k < nsel; ++k) {
        int32_t& g = dense_slots_[codes[sel[k]]];
        if (g < 0) {
          Key key;
          key.v[0] = codes[sel[k]];
          g = static_cast<int32_t>(insert(key));
        }
        gids[k] = static_cast<uint32_t>(g);
      }
      return;
    }
    for (size_t k = 0; k < nsel; ++k) gids[k] = find_or_insert(make_key(start + sel[k]));
  }

  void merge_from(GroupTable& o) {
    for (size_t g = 0; g < o.size(); ++g) {
      uint32_t mine = find_or_insert(o.key(g));
      Acc* dst = accs(mine);
      const Acc* src = o.accs(g);
      for (size_t a = 0; a < num_aggs_; ++a) dst[a].merge(src[a]);
    }
  }

 private:
  size_t insert(const Key& k) {
    groups_.push_back(k);
    accs_.resize(accs_.size() + num_aggs_);
    return groups_.size() - 1;
  }

  void grow() {
    std::vector<int32_t> old(slots_.size() * 2, -1);
    old.swap(slots_);
    size_t mask = slots_.size() - 1;
    for (size_t g = 0; g < groups_.size(); ++g) {
      size_t i = hash_key(groups_[g]) & mask;
      while (slots_[i] >= 0) i = (i + 1) & mask;
      slots_[i] = static_cast<int32_t>(g);
    }
  }

  std::vector<const Column*> keys_;
  size_t num_aggs_;
  bool dense_ = false;
  std::vector<int32_t> dense_slots_;
  std::vector<int32_t> slots_;
  std::vector<Key> groups_;
  std::vector<Acc> accs_;
};

// Row-at-a-time accumulation into per-group states, one tight loop per
// aggregate so the inner loop is branch-light and type-specialized.
void update_grouped(const AggSpec& a, size_t ai, GroupTable& gt, size_t start, const uint32_t* sel, size_t nsel,
                    const uint32_t* gids) {
  const Column* c = a.col;
  const bool nulls = c != nullptr && c->has_nulls();
  auto acc_of = [&](size_t k) -> Acc& { return gt.accs(gids[k])[ai]; };
  if (c == nullptr) {
    for (size_t k = 0; k < nsel; ++k) acc_of(k).count++;
    return;
  }
  auto valid = [&](size_t row) { return !nulls || c->valid[row] != 0; };
  const bool is_int = c->type == Type::Int64;
  switch (a.func) {
    case AggFunc::Count:
      for (size_t k = 0; k < nsel; ++k) acc_of(k).count += valid(start + sel[k]) ? 1 : 0;
      break;
    case AggFunc::Sum:
    case AggFunc::Avg:
      if (is_int) {
        for (size_t k = 0; k < nsel; ++k) {
          size_t row = start + sel[k];
          if (!valid(row)) continue;
          Acc& acc = acc_of(k);
          acc.count++;
          acc.isum += static_cast<uint64_t>(c->i64[row]);
        }
      } else {
        for (size_t k = 0; k < nsel; ++k) {
          size_t row = start + sel[k];
          if (!valid(row)) continue;
          Acc& acc = acc_of(k);
          acc.count++;
          acc.fsum += c->f64[row];
        }
      }
      break;
    case AggFunc::Min:
    case AggFunc::Max: {
      const bool is_min = a.func == AggFunc::Min;
      for (size_t k = 0; k < nsel; ++k) {
        size_t row = start + sel[k];
        if (!valid(row)) continue;
        Acc& acc = acc_of(k);
        acc.has = true;
        if (is_int) {
          int64_t v = c->i64[row];
          if (is_min)
            acc.imin = std::min(acc.imin, v);
          else
            acc.imax = std::max(acc.imax, v);
        } else {
          double v = c->f64[row];
          if (is_min && v < acc.fmin) acc.fmin = v;
          if (!is_min && v > acc.fmax) acc.fmax = v;
        }
      }
      break;
    }
  }
}

Value cell(const Column& c, size_t row) {
  if (!c.is_valid(row)) return std::monostate{};
  switch (c.type) {
    case Type::Int64: return c.i64[row];
    case Type::Float64: return c.f64[row];
    case Type::String: return c.dict[c.codes[row]];
  }
  return std::monostate{};
}

Value key_value(const Column& c, const Key& k, size_t i) {
  if (k.nulls & (1u << i)) return std::monostate{};
  switch (c.type) {
    case Type::Int64: return k.v[i];
    case Type::String: return c.dict[static_cast<size_t>(k.v[i])];
    case Type::Float64: {
      double d;
      std::memcpy(&d, &k.v[i], sizeof d);
      return d;
    }
  }
  return std::monostate{};
}

// ---------------------------------------------------------------------------
// Query plumbing

struct Plan {
  const Table* t = nullptr;
  std::unique_ptr<BPred> where;
  size_t pred_nodes = 0;
  size_t batch = 4096;
  int threads = 1;

  Scratch make_scratch() const {
    Scratch s;
    s.node.assign(pred_nodes, std::vector<uint8_t>(batch));
    s.mask.resize(batch);
    s.tmp.resize(batch);
    s.sel.resize(batch);
    s.gids.resize(batch);
    return s;
  }

  [[nodiscard]] size_t batches() const { return (t->num_rows + batch - 1) / batch; }

  // Fills s.mask for batch `b`; returns {start, n}.
  std::pair<size_t, size_t> filter(size_t b, Scratch& s) const {
    size_t start = b * batch;
    size_t n = std::min(batch, t->num_rows - start);
    if (where) {
      eval(*where, start, n, s.mask.data(), s);
    } else {
      kernels::mask_fill(s.mask.data(), n, true);
    }
    return {start, n};
  }
};

bool same_ref(const SelectItem& a, const SelectItem& b) {
  return a.kind == b.kind && (a.kind != SelectItem::Kind::Agg || a.func == b.func) && iequals(a.column, b.column);
}

Result run_aggregate(const Plan& plan, const sql::Select& q) {
  const Table& t = *plan.t;
  std::vector<const Column*> keys;
  for (const auto& g : q.group_by) keys.push_back(&column_of(t, g));
  if (keys.size() > kMaxKeys) fail("at most 4 GROUP BY columns are supported");

  std::vector<AggSpec> aggs;
  // For each select item: index into aggs, or -(1 + key index) for a key.
  std::vector<int> item_slot;
  for (const auto& it : q.items) {
    if (it.kind == SelectItem::Kind::Star) fail("SELECT * cannot be combined with aggregates or GROUP BY");
    if (it.kind == SelectItem::Kind::Column) {
      int ki = -1;
      for (size_t i = 0; i < q.group_by.size(); ++i) {
        if (iequals(q.group_by[i], it.column)) ki = static_cast<int>(i);
      }
      if (ki < 0) fail("column '" + it.column + "' must appear in GROUP BY or be aggregated");
      item_slot.push_back(-(1 + ki));
      continue;
    }
    AggSpec a{it.func, it.column.empty() ? nullptr : &column_of(t, it.column)};
    if (a.col != nullptr && a.col->type == Type::String && a.func != AggFunc::Count) {
      fail(std::string(sql::agg_name(a.func)) + " is not supported on STRING column '" + a.col->name + "'");
    }
    item_slot.push_back(static_cast<int>(aggs.size()));
    aggs.push_back(a);
  }

  const int workers = std::max(1, std::min<int>(plan.threads, static_cast<int>(std::max<size_t>(1, plan.batches()))));
  std::vector<Scratch> scratch;
  for (int w = 0; w < workers; ++w) scratch.push_back(plan.make_scratch());

  Result res;
  for (const auto& it : q.items) res.names.push_back(it.alias.empty() ? it.text : it.alias);

  if (keys.empty()) {
    std::vector<std::vector<Acc>> accs(static_cast<size_t>(workers), std::vector<Acc>(aggs.size()));
    parallel_for(plan.batches(), workers, [&](size_t b, int w) {
      Scratch& s = scratch[static_cast<size_t>(w)];
      auto [start, n] = plan.filter(b, s);
      for (size_t a = 0; a < aggs.size(); ++a)
        update_masked(aggs[a], accs[static_cast<size_t>(w)][a], start, n, s.mask.data(), s);
    });
    for (size_t w = 1; w < accs.size(); ++w) {
      for (size_t a = 0; a < aggs.size(); ++a) accs[0][a].merge(accs[w][a]);
    }
    std::vector<Value> row;
    for (int slot : item_slot)
      row.push_back(finalize(aggs[static_cast<size_t>(slot)], accs[0][static_cast<size_t>(slot)]));
    res.rows.push_back(std::move(row));
    return res;
  }

  std::vector<GroupTable> tables;
  for (int w = 0; w < workers; ++w) tables.emplace_back(keys, aggs.size());
  parallel_for(plan.batches(), workers, [&](size_t b, int w) {
    Scratch& s = scratch[static_cast<size_t>(w)];
    GroupTable& gt = tables[static_cast<size_t>(w)];
    auto [start, n] = plan.filter(b, s);
    size_t nsel = kernels::mask_to_sel(s.mask.data(), n, s.sel.data());
    if (nsel == 0) return;
    gt.lookup_batch(start, s.sel.data(), nsel, s.gids.data());
    for (size_t a = 0; a < aggs.size(); ++a) update_grouped(aggs[a], a, gt, start, s.sel.data(), nsel, s.gids.data());
  });
  for (size_t w = 1; w < tables.size(); ++w) tables[0].merge_from(tables[w]);

  GroupTable& gt = tables[0];
  res.rows.reserve(gt.size());
  for (size_t g = 0; g < gt.size(); ++g) {
    std::vector<Value> row;
    row.reserve(item_slot.size());
    for (int slot : item_slot) {
      if (slot < 0) {
        auto ki = static_cast<size_t>(-(slot + 1));
        row.push_back(key_value(*keys[ki], gt.key(g), ki));
      } else {
        row.push_back(finalize(aggs[static_cast<size_t>(slot)], gt.accs(g)[slot]));
      }
    }
    res.rows.push_back(std::move(row));
  }
  return res;
}

Result run_projection(const Plan& plan, const sql::Select& q, std::vector<SelectItem>& expanded) {
  const Table& t = *plan.t;
  std::vector<const Column*> cols;
  for (const auto& it : q.items) {
    if (it.kind == SelectItem::Kind::Star) {
      for (const auto& c : t.columns) {
        cols.push_back(&c);
        SelectItem e;
        e.kind = SelectItem::Kind::Column;
        e.column = e.text = c.name;
        expanded.push_back(e);
      }
    } else {
      cols.push_back(&column_of(t, it.column));
      expanded.push_back(it);
    }
  }
  Result res;
  for (const auto& e : expanded) res.names.push_back(e.alias.empty() ? e.text : e.alias);

  // Without ORDER BY a LIMIT lets us stop materializing early.
  const size_t cap = (q.order_by.empty() && q.limit) ? static_cast<size_t>(std::max<int64_t>(0, *q.limit))
                                                     : std::numeric_limits<size_t>::max();
  std::vector<uint32_t> rows_out;
  auto emit = [&](size_t row) {
    std::vector<Value> r;
    r.reserve(cols.size());
    for (const Column* c : cols) r.push_back(cell(*c, row));
    res.rows.push_back(std::move(r));
  };

  if (!plan.where) {
    for (size_t row = 0; row < t.num_rows && res.rows.size() < cap; ++row) emit(row);
    return res;
  }
  // Filter batches in parallel, then materialize in row order.
  std::vector<std::vector<uint32_t>> selected(plan.batches());
  const int workers = std::max(1, std::min<int>(plan.threads, static_cast<int>(std::max<size_t>(1, plan.batches()))));
  std::vector<Scratch> scratch;
  for (int w = 0; w < workers; ++w) scratch.push_back(plan.make_scratch());
  parallel_for(plan.batches(), workers, [&](size_t b, int w) {
    Scratch& s = scratch[static_cast<size_t>(w)];
    auto [start, n] = plan.filter(b, s);
    size_t nsel = kernels::mask_to_sel(s.mask.data(), n, s.sel.data());
    selected[b].assign(s.sel.begin(), s.sel.begin() + static_cast<ptrdiff_t>(nsel));
  });
  for (size_t b = 0; b < selected.size() && res.rows.size() < cap; ++b) {
    for (uint32_t i : selected[b]) {
      if (res.rows.size() >= cap) break;
      emit(b * plan.batch + i);
    }
  }
  return res;
}

void apply_order_limit(Result& res, const sql::Select& q, const std::vector<SelectItem>& items) {
  if (!q.order_by.empty()) {
    std::vector<std::pair<size_t, bool>> keys;
    for (const auto& o : q.order_by) {
      size_t idx = items.size();
      if (o.ordinal) {
        if (static_cast<size_t>(*o.ordinal) > items.size()) fail("ORDER BY ordinal out of range");
        idx = static_cast<size_t>(*o.ordinal - 1);
      } else {
        for (size_t i = 0; i < items.size() && idx == items.size(); ++i) {
          if (o.key.kind == SelectItem::Kind::Column && !items[i].alias.empty() &&
              iequals(items[i].alias, o.key.column))
            idx = i;
        }
        for (size_t i = 0; i < items.size() && idx == items.size(); ++i) {
          if (same_ref(items[i], o.key)) idx = i;
        }
      }
      if (idx == items.size()) fail("ORDER BY key '" + o.key.text + "' must appear in the SELECT list");
      keys.emplace_back(idx, o.desc);
    }
    auto less = [&](const std::vector<Value>& a, const std::vector<Value>& b) {
      for (auto [i, desc] : keys) {
        int c = compare_values(a[i], b[i]);
        if (c != 0) return desc ? c > 0 : c < 0;
      }
      return false;
    };
    if (q.limit && static_cast<size_t>(std::max<int64_t>(0, *q.limit)) < res.rows.size()) {
      auto k = static_cast<ptrdiff_t>(*q.limit);
      std::partial_sort(res.rows.begin(), res.rows.begin() + k, res.rows.end(), less);
    } else {
      std::stable_sort(res.rows.begin(), res.rows.end(), less);
    }
  }
  if (q.limit) {
    auto k = static_cast<size_t>(std::max<int64_t>(0, *q.limit));
    if (res.rows.size() > k) res.rows.resize(k);
  }
}

std::string fmt_double(double d, bool roundtrip) {
  char buf[64];
  if (roundtrip) {
    auto r = std::to_chars(buf, buf + sizeof buf, d);
    return std::string(buf, r.ptr);
  }
  std::snprintf(buf, sizeof buf, "%.10g", d);
  return buf;
}

std::string fmt(const Value& v, bool roundtrip) {
  struct V {
    bool rt;
    std::string operator()(std::monostate) const { return "NULL"; }
    std::string operator()(int64_t i) const { return std::to_string(i); }
    std::string operator()(double d) const { return fmt_double(d, rt); }
    std::string operator()(const std::string& s) const { return s; }
  };
  return std::visit(V{roundtrip}, v);
}

}  // namespace

int compare_values(const Value& a, const Value& b) {
  auto rank = [](const Value& v) { return v.index() == 0 ? 0 : (v.index() == 3 ? 2 : 1); };
  int ra = rank(a), rb = rank(b);
  if (ra != rb) return ra < rb ? -1 : 1;
  if (ra == 0) return 0;
  if (ra == 2) {
    int c = std::get<3>(a).compare(std::get<3>(b));
    return c < 0 ? -1 : (c > 0 ? 1 : 0);
  }
  if (a.index() == 1 && b.index() == 1) {
    int64_t x = std::get<1>(a), y = std::get<1>(b);
    return x < y ? -1 : (x > y ? 1 : 0);
  }
  double x = a.index() == 1 ? static_cast<double>(std::get<1>(a)) : std::get<2>(a);
  double y = b.index() == 1 ? static_cast<double>(std::get<1>(b)) : std::get<2>(b);
  return x < y ? -1 : (x > y ? 1 : 0);
}

Result execute(const Table& t, const sql::Select& q, const ExecOptions& opts) {
  Plan plan;
  plan.t = &t;
  plan.batch = std::max<size_t>(16, opts.batch_rows);
  plan.threads = opts.threads > 0 ? opts.threads : default_threads();
  if (q.where) plan.where = bind(t, *q.where, plan.pred_nodes);

  bool aggregate = !q.group_by.empty();
  for (const auto& it : q.items) aggregate |= it.kind == SelectItem::Kind::Agg;

  Result res;
  std::vector<SelectItem> items;
  if (aggregate) {
    res = run_aggregate(plan, q);
    items = q.items;
  } else {
    res = run_projection(plan, q, items);
  }
  apply_order_limit(res, q, items);
  return res;
}

Result query(const Table& t, const std::string& sql_text, const ExecOptions& opts) {
  return execute(t, sql::parse(sql_text), opts);
}

std::string format_value(const Value& v) { return fmt(v, false); }

std::string format_table(const Result& r) {
  const size_t nc = r.names.size();
  std::vector<std::vector<std::string>> cells;
  std::vector<size_t> width(nc, 0);
  std::vector<bool> numeric(nc, true);
  for (size_t c = 0; c < nc; ++c) width[c] = r.names[c].size();
  for (const auto& row : r.rows) {
    std::vector<std::string> out;
    for (size_t c = 0; c < nc; ++c) {
      out.push_back(fmt(row[c], false));
      width[c] = std::max(width[c], out.back().size());
      if (row[c].index() == 3) numeric[c] = false;
    }
    cells.push_back(std::move(out));
  }
  std::string s;
  auto rule = [&] {
    s += '+';
    for (size_t c = 0; c < nc; ++c) s += std::string(width[c] + 2, '-') + '+';
    s += '\n';
  };
  auto line = [&](const std::vector<std::string>& v, bool header) {
    s += '|';
    for (size_t c = 0; c < nc; ++c) {
      std::string pad(width[c] - v[c].size(), ' ');
      s += ' ';
      s += (numeric[c] && !header) ? pad + v[c] : v[c] + pad;
      s += " |";
    }
    s += '\n';
  };
  rule();
  line(r.names, true);
  rule();
  for (const auto& row : cells) line(row, false);
  rule();
  s += std::to_string(r.rows.size()) + (r.rows.size() == 1 ? " row\n" : " rows\n");
  return s;
}

std::string format_csv(const Result& r) {
  auto quote = [](const std::string& v) {
    if (v.find_first_of(",\"\n\r") == std::string::npos) return v;
    std::string q = "\"";
    for (char c : v) {
      if (c == '"') q += '"';
      q += c;
    }
    return q + "\"";
  };
  std::string s;
  for (size_t c = 0; c < r.names.size(); ++c) s += (c ? "," : "") + quote(r.names[c]);
  s += '\n';
  for (const auto& row : r.rows) {
    for (size_t c = 0; c < row.size(); ++c) {
      if (c) s += ',';
      if (row[c].index() != 0) s += quote(fmt(row[c], true));
    }
    s += '\n';
  }
  return s;
}

}  // namespace vecq
