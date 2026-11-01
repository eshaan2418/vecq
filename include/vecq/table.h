#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vecq {

enum class Type : uint8_t { Int64, Float64, String };

const char* type_name(Type t);

// A single in-memory column. Exactly one of i64 / f64 / codes is populated,
// depending on `type`. String columns are dictionary encoded: `codes[i]`
// indexes into `dict`.
//
// `valid` is a byte mask (0xFF = present, 0x00 = NULL). It is empty when the
// column has no NULLs, which lets the executor skip validity handling.
struct Column {
  std::string name;
  Type type = Type::Int64;
  std::vector<int64_t> i64;
  std::vector<double> f64;
  std::vector<uint32_t> codes;
  std::vector<std::string> dict;
  std::vector<uint8_t> valid;

  [[nodiscard]] size_t size() const;
  [[nodiscard]] bool has_nulls() const { return !valid.empty(); }
  [[nodiscard]] bool is_valid(size_t i) const { return valid.empty() || valid[i] != 0; }
};

struct Table {
  std::vector<Column> columns;
  size_t num_rows = 0;

  // Case-insensitive lookup. Returns -1 if absent.
  [[nodiscard]] int find(std::string_view name) const;
};

bool iequals(std::string_view a, std::string_view b);

}  // namespace vecq
