#include "vecq/table.h"

#include <cctype>

namespace vecq {

const char* type_name(Type t) {
  switch (t) {
    case Type::Int64: return "INT64";
    case Type::Float64: return "FLOAT64";
    case Type::String: return "STRING";
  }
  return "?";
}

size_t Column::size() const {
  switch (type) {
    case Type::Int64: return i64.size();
    case Type::Float64: return f64.size();
    case Type::String: return codes.size();
  }
  return 0;
}

bool iequals(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) {
      return false;
    }
  }
  return true;
}

int Table::find(std::string_view name) const {
  for (size_t i = 0; i < columns.size(); ++i) {
    if (iequals(columns[i].name, name)) return static_cast<int>(i);
  }
  return -1;
}

}  // namespace vecq
