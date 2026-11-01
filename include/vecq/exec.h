#pragma once

#include <string>
#include <variant>
#include <vector>

#include "vecq/sql.h"
#include "vecq/table.h"

namespace vecq {

using Value = std::variant<std::monostate, int64_t, double, std::string>;

struct Result {
  std::vector<std::string> names;
  std::vector<std::vector<Value>> rows;
};

struct ExecOptions {
  int threads = 0;           // 0 = hardware concurrency
  size_t batch_rows = 4096;  // rows per vector batch (and per morsel)
};

// Executes a parsed query. Throws std::runtime_error for semantic errors
// (unknown column, type mismatch, non-grouped column in an aggregate, ...).
Result execute(const Table& t, const sql::Select& q, const ExecOptions& opts = {});
Result query(const Table& t, const std::string& sql, const ExecOptions& opts = {});

// SQL ordering for result values: NULL < numbers < strings; ints and doubles
// compare numerically.
int compare_values(const Value& a, const Value& b);

std::string format_value(const Value& v);
std::string format_table(const Result& r);
std::string format_csv(const Result& r);

}  // namespace vecq
