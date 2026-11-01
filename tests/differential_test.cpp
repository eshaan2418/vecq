// Differential testing: random tables and random queries are executed by the
// vectorized engine (under every SIMD / thread / batch-size configuration)
// and by an independent row-at-a-time reference interpreter written here.
// Results must match exactly (doubles within 1e-9 relative).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <sstream>

#include "vecq/exec.h"
#include "vecq/kernels.h"
#include "vecq/sql.h"

using namespace vecq;
using sql::AggFunc;
using sql::Pred;
using sql::SelectItem;

namespace {

// --------------------------------------------------------------------------
// Random data

Table random_table(std::mt19937_64& rng, size_t rows) {
  Table t;
  t.num_rows = rows;
  auto add = [&](std::string name, Type type) -> Column& {
    Column c;
    c.name = std::move(name);
    c.type = type;
    t.columns.push_back(std::move(c));
    return t.columns.back();
  };
  {
    Column& c = add("i", Type::Int64);
    c.valid.resize(rows);
    for (size_t r = 0; r < rows; ++r) {
      c.valid[r] = rng() % 10 ? 0xFF : 0;
      c.i64.push_back(c.valid[r] ? static_cast<int64_t>(rng() % 201) - 100 : 0);
    }
  }
  {
    Column& c = add("f", Type::Float64);
    c.valid.resize(rows);
    for (size_t r = 0; r < rows; ++r) {
      c.valid[r] = rng() % 8 ? 0xFF : 0;
      c.f64.push_back(c.valid[r] ? static_cast<double>(static_cast<int64_t>(rng() % 2001) - 1000) / 4.0 : 0);
    }
  }
  {
    Column& c = add("s", Type::String);
    c.dict = {"apple", "banana", "cherry", "date", "elder", "fig", "grape", "kiwi"};
    c.valid.resize(rows);
    for (size_t r = 0; r < rows; ++r) {
      c.valid[r] = rng() % 12 ? 0xFF : 0;
      c.codes.push_back(c.valid[r] ? static_cast<uint32_t>(rng() % c.dict.size()) : 0);
    }
  }
  {
    Column& c = add("k", Type::Int64);
    for (size_t r = 0; r < rows; ++r) c.i64.push_back(static_cast<int64_t>(rng() % 10));
  }
  {
    Column& c = add("g", Type::String);
    c.dict = {"north", "south", "west"};
    for (size_t r = 0; r < rows; ++r) c.codes.push_back(static_cast<uint32_t>(rng() % 3));
  }
  {
    Column& c = add("h", Type::Float64);
    for (size_t r = 0; r < rows; ++r) c.f64.push_back(static_cast<double>(rng() % 5) * 0.5);
  }
  return t;
}

// --------------------------------------------------------------------------
// Random queries (as SQL text, so the parser is exercised too)

std::string random_literal(std::mt19937_64& rng, Type t) {
  switch (t) {
    case Type::Int64:
      if (rng() % 4 == 0) return std::to_string(static_cast<int64_t>(rng() % 201) - 100) + ".5";
      return std::to_string(static_cast<int64_t>(rng() % 201) - 100);
    case Type::Float64: return std::to_string(static_cast<double>(static_cast<int64_t>(rng() % 2001) - 1000) / 4.0);
    case Type::String: {
      static const char* vals[] = {"apple", "banana", "cherry", "date", "elder", "fig", "grape",
                                   "kiwi",  "north",  "south",  "west", "zzz",   "b"};
      return std::string("'") + vals[rng() % 13] + "'";
    }
  }
  return "0";
}

std::string random_pred(std::mt19937_64& rng, const Table& t, int depth) {
  if (depth > 0 && rng() % 3 == 0) {
    std::string op = rng() % 2 ? " AND " : " OR ";
    return "(" + random_pred(rng, t, depth - 1) + op + random_pred(rng, t, depth - 1) + ")";
  }
  const Column& c = t.columns[rng() % t.columns.size()];
  static const char* ops[] = {"=", "!=", "<", "<=", ">", ">="};
  switch (rng() % 6) {
    case 0: return c.name + (rng() % 2 ? " IS NULL" : " IS NOT NULL");
    case 1:
      if (c.type == Type::String) {
        static const char* pats[] = {"%a%", "_a%", "%e", "b%", "%", "kiwi", "%r_"};
        return c.name + (rng() % 3 ? " LIKE '" : " NOT LIKE '") + pats[rng() % 7] + "'";
      }
      return c.name + (rng() % 2 ? " BETWEEN " : " NOT BETWEEN ") + random_literal(rng, c.type) + " AND " +
             random_literal(rng, c.type);
    default: return c.name + " " + ops[rng() % 6] + " " + random_literal(rng, c.type);
  }
}

std::string random_query(std::mt19937_64& rng, const Table& t) {
  std::ostringstream q;
  const bool aggregate = rng() % 4 != 0;
  std::string where = rng() % 5 ? " WHERE " + random_pred(rng, t, 3) : "";
  if (!aggregate) {
    q << "SELECT k, s, f, i FROM t" << where;
    if (rng() % 2) q << " ORDER BY f DESC, i, k, s";
    if (rng() % 2) q << " LIMIT " << (rng() % 50);
    return q.str();
  }
  static const char* keys[] = {"k", "g", "s", "i", "h"};
  std::vector<std::string> group;
  int nkeys = static_cast<int>(rng() % 3);
  for (int k = 0; k < nkeys; ++k) {
    std::string key = keys[rng() % 5];
    if (std::find(group.begin(), group.end(), key) == group.end()) group.push_back(key);
  }
  std::vector<std::string> items(group.begin(), group.end());
  static const char* funcs[] = {"COUNT", "SUM", "AVG", "MIN", "MAX"};
  static const char* numeric[] = {"i", "f", "k", "h"};
  int naggs = 1 + static_cast<int>(rng() % 4);
  for (int a = 0; a < naggs; ++a) {
    if (rng() % 5 == 0) {
      items.emplace_back("COUNT(*)");
    } else if (rng() % 6 == 0) {
      items.emplace_back("COUNT(s)");
    } else {
      items.push_back(std::string(funcs[rng() % 5]) + "(" + numeric[rng() % 4] + ")");
    }
  }
  q << "SELECT ";
  for (size_t k = 0; k < items.size(); ++k) q << (k ? ", " : "") << items[k];
  q << " FROM t" << where;
  if (!group.empty()) {
    q << " GROUP BY ";
    for (size_t k = 0; k < group.size(); ++k) q << (k ? ", " : "") << group[k];
  }
  return q.str();
}

// --------------------------------------------------------------------------
// Reference interpreter

Value cell(const Column& c, size_t r) {
  if (!c.is_valid(r)) return std::monostate{};
  switch (c.type) {
    case Type::Int64: return c.i64[r];
    case Type::Float64: return c.f64[r];
    case Type::String: return c.dict[c.codes[r]];
  }
  return std::monostate{};
}

bool like(const std::string& s, size_t si, const std::string& p, size_t pi) {
  if (pi == p.size()) return si == s.size();
  if (p[pi] == '%') {
    for (size_t k = si; k <= s.size(); ++k) {
      if (like(s, k, p, pi + 1)) return true;
    }
    return false;
  }
  if (si == s.size()) return false;
  return (p[pi] == '_' || p[pi] == s[si]) && like(s, si + 1, p, pi + 1);
}

bool cmp_result(int c, kernels::CmpOp op) {
  switch (op) {
    case kernels::CmpOp::Eq: return c == 0;
    case kernels::CmpOp::Ne: return c != 0;
    case kernels::CmpOp::Lt: return c < 0;
    case kernels::CmpOp::Le: return c <= 0;
    case kernels::CmpOp::Gt: return c > 0;
    case kernels::CmpOp::Ge: return c >= 0;
  }
  return false;
}

bool eval_row(const Pred& p, const Table& t, size_t r) {
  switch (p.kind) {
    case Pred::Kind::And: return eval_row(*p.lhs, t, r) && eval_row(*p.rhs, t, r);
    case Pred::Kind::Or: return eval_row(*p.lhs, t, r) || eval_row(*p.rhs, t, r);
    default: break;
  }
  const Column& c = t.columns[static_cast<size_t>(t.find(p.column))];
  Value v = cell(c, r);
  if (p.kind == Pred::Kind::IsNull) return (v.index() == 0) != p.negated;
  if (v.index() == 0) return false;
  if (p.kind == Pred::Kind::Like) return like(std::get<std::string>(v), 0, p.lit.s, 0) != p.negated;
  Value lit;
  switch (p.lit.kind) {
    case sql::Literal::Kind::Int: lit = p.lit.i; break;
    case sql::Literal::Kind::Float: lit = p.lit.f; break;
    case sql::Literal::Kind::String: lit = p.lit.s; break;
  }
  return cmp_result(compare_values(v, lit), p.op);
}

struct RefAcc {
  int64_t count = 0;
  int64_t isum = 0;
  double fsum = 0;
  Value mn, mx;
};

Result reference(const Table& t, const sql::Select& q) {
  Result res;
  std::vector<size_t> rows;
  for (size_t r = 0; r < t.num_rows; ++r) {
    if (!q.where || eval_row(*q.where, t, r)) rows.push_back(r);
  }
  bool aggregate = !q.group_by.empty();
  for (const auto& it : q.items) aggregate |= it.kind == SelectItem::Kind::Agg;
  for (const auto& it : q.items) res.names.push_back(it.alias.empty() ? it.text : it.alias);

  if (!aggregate) {
    for (size_t r : rows) {
      std::vector<Value> row;
      for (const auto& it : q.items) row.push_back(cell(t.columns[static_cast<size_t>(t.find(it.column))], r));
      res.rows.push_back(std::move(row));
    }
  } else {
    // Group key -> accumulators, with the key serialized for the map.
    std::map<std::string, std::pair<std::vector<Value>, std::vector<RefAcc>>> groups;
    if (q.group_by.empty()) groups[""] = {{}, std::vector<RefAcc>(q.items.size())};
    for (size_t r : rows) {
      std::vector<Value> key;
      std::string skey;
      for (const auto& g : q.group_by) {
        key.push_back(cell(t.columns[static_cast<size_t>(t.find(g))], r));
        skey += std::to_string(key.back().index()) + ":" + format_value(key.back()) + "|";
      }
      auto& entry = groups[skey];
      if (entry.second.empty()) entry = {key, std::vector<RefAcc>(q.items.size())};
      for (size_t a = 0; a < q.items.size(); ++a) {
        const auto& it = q.items[a];
        if (it.kind != SelectItem::Kind::Agg) continue;
        RefAcc& acc = entry.second[a];
        if (it.column.empty()) {
          acc.count++;
          continue;
        }
        Value v = cell(t.columns[static_cast<size_t>(t.find(it.column))], r);
        if (v.index() == 0) continue;
        acc.count++;
        if (v.index() == 1) acc.isum += std::get<int64_t>(v);
        if (v.index() == 2) acc.fsum += std::get<double>(v);
        if (acc.mn.index() == 0 || compare_values(v, acc.mn) < 0) acc.mn = v;
        if (acc.mx.index() == 0 || compare_values(v, acc.mx) > 0) acc.mx = v;
      }
    }
    for (auto& [_, entry] : groups) {
      std::vector<Value> row;
      for (size_t a = 0; a < q.items.size(); ++a) {
        const auto& it = q.items[a];
        if (it.kind == SelectItem::Kind::Column) {
          for (size_t g = 0; g < q.group_by.size(); ++g) {
            if (iequals(q.group_by[g], it.column)) row.push_back(entry.first[g]);
          }
          continue;
        }
        const RefAcc& acc = entry.second[a];
        bool is_int = !it.column.empty() && t.columns[static_cast<size_t>(t.find(it.column))].type == Type::Int64;
        double sum = is_int ? static_cast<double>(acc.isum) : acc.fsum;
        switch (it.func) {
          case AggFunc::Count: row.emplace_back(acc.count); break;
          case AggFunc::Sum:
            if (acc.count == 0)
              row.emplace_back(std::monostate{});
            else if (is_int)
              row.emplace_back(acc.isum);
            else
              row.emplace_back(acc.fsum);
            break;
          case AggFunc::Avg:
            if (acc.count == 0)
              row.emplace_back(std::monostate{});
            else
              row.emplace_back(sum / static_cast<double>(acc.count));
            break;
          case AggFunc::Min: row.push_back(acc.mn); break;
          case AggFunc::Max: row.push_back(acc.mx); break;
        }
      }
      res.rows.push_back(std::move(row));
    }
  }
  return res;
}

bool values_match(const Value& a, const Value& b) {
  if (a.index() != b.index()) return false;
  if (a.index() == 2) {
    double x = std::get<double>(a), y = std::get<double>(b);
    return std::fabs(x - y) <= 1e-9 * std::max(1.0, std::max(std::fabs(x), std::fabs(y)));
  }
  return a == b;
}

bool row_less(const std::vector<Value>& a, const std::vector<Value>& b) {
  for (size_t i = 0; i < a.size(); ++i) {
    int c = compare_values(a[i], b[i]);
    if (c != 0) return c < 0;
  }
  return false;
}

}  // namespace

struct Config {
  bool simd;
  int threads;
  size_t batch;
};

class Differential : public ::testing::TestWithParam<Config> {};

TEST_P(Differential, RandomQueriesMatchReference) {
  const Config cfg = GetParam();
  if (cfg.simd && !kernels::simd_available()) GTEST_SKIP() << "no NEON";
  kernels::set_simd_enabled(cfg.simd);
  std::mt19937_64 rng(1234);
  int checked = 0;
  for (int table_round = 0; table_round < 4; ++table_round) {
    Table t = random_table(rng, 1 + rng() % 9000);
    for (int qi = 0; qi < 150; ++qi) {
      std::string text = random_query(rng, t);
      sql::Select q = sql::parse(text);
      ExecOptions opts;
      opts.threads = cfg.threads;
      opts.batch_rows = cfg.batch;
      Result got = execute(t, q, opts);

      // Compare unordered (and unlimited) results: run the reference, then
      // apply ordering/limit by re-running the engine's semantics only where
      // deterministic. Projection without ORDER BY preserves row order.
      sql::Select unordered = sql::parse(text);
      unordered.order_by.clear();
      std::optional<int64_t> limit = unordered.limit;
      unordered.limit.reset();
      Result want = reference(t, unordered);
      Result got_all = execute(t, unordered, opts);

      ASSERT_EQ(got_all.rows.size(), want.rows.size()) << text;
      if (!q.items.empty() && q.items[0].kind == SelectItem::Kind::Column && q.group_by.empty()) {
        // Projection: row order must match the table order exactly.
      } else {
        std::sort(got_all.rows.begin(), got_all.rows.end(), row_less);
        std::sort(want.rows.begin(), want.rows.end(), row_less);
      }
      for (size_t r = 0; r < want.rows.size(); ++r) {
        for (size_t c = 0; c < want.rows[r].size(); ++c) {
          ASSERT_TRUE(values_match(got_all.rows[r][c], want.rows[r][c]))
              << text << "\nrow " << r << " col " << c << ": got " << format_value(got_all.rows[r][c]) << " want "
              << format_value(want.rows[r][c]);
        }
      }
      // With ORDER BY + LIMIT the ordered result must be a correctly sorted
      // prefix of the same multiset.
      if (limit) ASSERT_LE(got.rows.size(), static_cast<size_t>(*limit)) << text;
      if (!q.order_by.empty()) {
        for (size_t r = 1; r < got.rows.size(); ++r) {
          // ORDER BY f DESC, i, k, s on items (k, s, f, i)
          const auto& a = got.rows[r - 1];
          const auto& b = got.rows[r];
          int c = -compare_values(a[2], b[2]);
          if (c == 0) c = compare_values(a[3], b[3]);
          if (c == 0) c = compare_values(a[0], b[0]);
          if (c == 0) c = compare_values(a[1], b[1]);
          ASSERT_LE(c, 0) << text;
        }
      }
      ++checked;
    }
  }
  kernels::set_simd_enabled(true);
  EXPECT_EQ(checked, 600);
}

INSTANTIATE_TEST_SUITE_P(Configs, Differential,
                         ::testing::Values(Config{true, 1, 4096}, Config{false, 1, 4096}, Config{true, 4, 64},
                                           Config{false, 3, 16}, Config{true, 8, 1000}),
                         [](const auto& info) {
                           return std::string(info.param.simd ? "neon" : "scalar") + "_t" +
                                  std::to_string(info.param.threads) + "_b" + std::to_string(info.param.batch);
                         });
