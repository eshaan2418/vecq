#pragma once

// A small SQL dialect:
//
//   SELECT item [, item]* FROM name
//     [WHERE pred]
//     [GROUP BY col [, col]*]
//     [ORDER BY key [ASC|DESC] [, ...]]
//     [LIMIT n]
//
//   item := * | col [AS alias] | agg '(' (col | *) ')' [AS alias]
//   agg  := COUNT | SUM | AVG | MIN | MAX
//   pred := pred OR pred | pred AND pred | '(' pred ')'
//         | col op literal | literal op col
//         | col [NOT] BETWEEN literal AND literal
//         | col [NOT] LIKE 'pattern' | col IS [NOT] NULL
//   op   := = | != | <> | < | <= | > | >=
//   key  := col | alias | agg '(' ... ')' | ordinal

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vecq/kernels.h"

namespace vecq::sql {

struct Literal {
  enum class Kind : uint8_t { Int, Float, String } kind = Kind::Int;
  int64_t i = 0;
  double f = 0;
  std::string s;
};

enum class AggFunc : uint8_t { Count, Sum, Avg, Min, Max };
const char* agg_name(AggFunc f);

struct SelectItem {
  enum class Kind : uint8_t { Star, Column, Agg } kind = Kind::Column;
  std::string column;  // empty for COUNT(*)
  AggFunc func = AggFunc::Count;
  std::string alias;
  // Canonical text, e.g. "SUM(fare)" or "payment"; used for headers and
  // ORDER BY matching.
  std::string text;
};

struct Pred {
  enum class Kind : uint8_t { Cmp, Like, IsNull, And, Or } kind = Kind::Cmp;
  std::string column;
  kernels::CmpOp op = kernels::CmpOp::Eq;
  Literal lit;
  bool negated = false;  // NOT LIKE, IS NOT NULL
  std::unique_ptr<Pred> lhs, rhs;
};

struct OrderItem {
  SelectItem key;              // column / agg reference
  std::optional<int> ordinal;  // 1-based
  bool desc = false;
};

struct Select {
  std::vector<SelectItem> items;
  std::string table;
  std::unique_ptr<Pred> where;
  std::vector<std::string> group_by;
  std::vector<OrderItem> order_by;
  std::optional<int64_t> limit;
};

// Throws std::runtime_error with a position-annotated message on bad input.
Select parse(const std::string& query);

}  // namespace vecq::sql
