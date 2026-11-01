#pragma once

#include <string>
#include <string_view>

#include "vecq/table.h"

namespace vecq {

struct CsvOptions {
  char delimiter = ',';
  bool header = true;
  int threads = 0;            // 0 = hardware concurrency
  size_t sample_rows = 4096;  // rows used for initial type inference
  size_t min_chunk_bytes = size_t{1} << 20;
};

// Parses RFC 4180 CSV (quoted fields, "" escapes, embedded newlines, CRLF).
// Column types are inferred from a sample and promoted (INT64 -> FLOAT64 ->
// STRING) with a re-parse if a later value does not fit. Empty unquoted
// fields are NULL. Throws std::runtime_error on malformed input.
Table parse_csv(std::string_view data, const CsvOptions& opts = {});

// mmaps `path` and calls parse_csv.
Table load_csv(const std::string& path, const CsvOptions& opts = {});

}  // namespace vecq
