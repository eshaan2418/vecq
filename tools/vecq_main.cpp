// vecq — run SQL directly over a CSV file.

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "vecq/csv.h"
#include "vecq/exec.h"
#include "vecq/kernels.h"

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: vecq [options] <file.csv> [\"SQL\"]\n"
               "\n"
               "Loads the CSV into memory (columnar, typed) and runs the query.\n"
               "Without a query, reads one query per line from stdin.\n"
               "The table name in FROM is arbitrary.\n"
               "\n"
               "options:\n"
               "  -t, --threads N   worker threads (default: all cores)\n"
               "      --no-simd     use portable kernels instead of NEON\n"
               "      --csv         print results as CSV\n"
               "      --timing      print load/query times to stderr\n"
               "      --repeat N    run each query N times (with --timing)\n"
               "      --schema      print the inferred schema and exit\n"
               "  -d, --delim C     field delimiter (default ',')\n"
               "      --no-header   first row is data\n");
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  vecq::CsvOptions copts;
  vecq::ExecOptions eopts;
  bool csv_out = false, timing = false, schema = false;
  int repeat = 1;
  std::string path, query;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "vecq: %s needs a value\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else if (a == "-t" || a == "--threads") {
      eopts.threads = copts.threads = std::atoi(need("--threads").c_str());
    } else if (a == "--no-simd") {
      vecq::kernels::set_simd_enabled(false);
    } else if (a == "--csv") {
      csv_out = true;
    } else if (a == "--timing") {
      timing = true;
    } else if (a == "--repeat") {
      repeat = std::max(1, std::atoi(need("--repeat").c_str()));
    } else if (a == "--schema") {
      schema = true;
    } else if (a == "-d" || a == "--delim") {
      std::string d = need("--delim");
      copts.delimiter = d == "\\t" ? '\t' : d[0];
    } else if (a == "--no-header") {
      copts.header = false;
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "vecq: unknown option %s\n", a.c_str());
      usage();
      return 2;
    } else if (path.empty()) {
      path = a;
    } else if (query.empty()) {
      query = a;
    } else {
      usage();
      return 2;
    }
  }
  if (path.empty()) {
    usage();
    return 2;
  }

  vecq::Table table;
  try {
    auto t0 = std::chrono::steady_clock::now();
    table = vecq::load_csv(path, copts);
    if (timing) {
      std::fprintf(stderr, "load: %.1f ms (%zu rows, %zu columns, simd=%s)\n", ms_since(t0), table.num_rows,
                   table.columns.size(), vecq::kernels::simd_enabled() ? "neon" : "off");
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vecq: %s\n", e.what());
    return 1;
  }

  if (schema) {
    for (const auto& c : table.columns) {
      std::printf("%-24s %-8s%s", c.name.c_str(), vecq::type_name(c.type), c.has_nulls() ? " nullable" : "");
      if (c.type == vecq::Type::String) std::printf(" (%zu distinct)", c.dict.size());
      std::printf("\n");
    }
    return 0;
  }

  auto run = [&](const std::string& q) -> bool {
    try {
      vecq::Result r;
      for (int k = 0; k < repeat; ++k) {
        auto t0 = std::chrono::steady_clock::now();
        r = vecq::query(table, q, eopts);
        if (timing) std::fprintf(stderr, "query: %.3f ms\n", ms_since(t0));
      }
      std::string out = csv_out ? vecq::format_csv(r) : vecq::format_table(r);
      std::fwrite(out.data(), 1, out.size(), stdout);
      std::fflush(stdout);
      return true;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return false;
    }
  };

  if (!query.empty()) return run(query) ? 0 : 1;

  const bool tty = isatty(STDIN_FILENO) != 0;
  std::string line;
  bool ok = true;
  for (;;) {
    if (tty) std::fputs("vecq> ", stdout), std::fflush(stdout);
    if (!std::getline(std::cin, line)) break;
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    if (line == "\\q" || line == "quit" || line == "exit") break;
    ok = run(line) && ok;
  }
  return ok ? 0 : 1;
}
