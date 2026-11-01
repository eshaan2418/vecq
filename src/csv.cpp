#include "vecq/csv.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <unordered_map>

#include "vecq/parallel.h"

namespace vecq {

namespace {

// ---------------------------------------------------------------------------
// Field tokenizer

struct Field {
  std::string_view text;
  bool quoted = false;
  bool end_of_row = false;
};

class Reader {
 public:
  Reader(std::string_view data, size_t pos, char delim) : data_(data), pos_(pos), delim_(delim) {}

  [[nodiscard]] size_t pos() const { return pos_; }
  [[nodiscard]] bool done() const { return pos_ >= data_.size(); }

  // Skips empty lines. Returns false at end of input.
  bool start_row() {
    while (pos_ < data_.size()) {
      char c = data_[pos_];
      if (c == '\n') {
        ++pos_;
      } else if (c == '\r' && pos_ + 1 < data_.size() && data_[pos_ + 1] == '\n') {
        pos_ += 2;
      } else {
        return true;
      }
    }
    return false;
  }

  Field next(std::deque<std::string>* owned) {
    Field f;
    const size_t n = data_.size();
    if (pos_ >= n) {
      f.end_of_row = true;
      return f;
    }
    if (data_[pos_] == '"') {
      f.quoted = true;
      size_t start = ++pos_;
      bool escaped = false;
      size_t end;
      for (;;) {
        const void* q = std::memchr(data_.data() + pos_, '"', n - pos_);
        if (q == nullptr) throw std::runtime_error("csv: unterminated quoted field");
        size_t qi = static_cast<size_t>(static_cast<const char*>(q) - data_.data());
        if (qi + 1 < n && data_[qi + 1] == '"') {
          escaped = true;
          pos_ = qi + 2;
          continue;
        }
        end = qi;
        pos_ = qi + 1;
        break;
      }
      std::string_view raw = data_.substr(start, end - start);
      if (escaped) {
        std::string s;
        s.reserve(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
          s.push_back(raw[i]);
          if (raw[i] == '"') ++i;  // skip the second quote of ""
        }
        owned->push_back(std::move(s));
        f.text = owned->back();
      } else {
        f.text = raw;
      }
      if (pos_ < n && data_[pos_] == '\r') ++pos_;
      if (pos_ >= n) {
        f.end_of_row = true;
      } else if (data_[pos_] == delim_) {
        ++pos_;
      } else if (data_[pos_] == '\n') {
        ++pos_;
        f.end_of_row = true;
      } else {
        throw std::runtime_error("csv: unexpected character after closing quote");
      }
      return f;
    }
    size_t e = pos_;
    while (e < n && data_[e] != delim_ && data_[e] != '\n') ++e;
    std::string_view text = data_.substr(pos_, e - pos_);
    if (e >= n) {
      f.end_of_row = true;
      pos_ = n;
    } else if (data_[e] == '\n') {
      f.end_of_row = true;
      pos_ = e + 1;
    } else {
      pos_ = e + 1;
    }
    if (f.end_of_row && !text.empty() && text.back() == '\r') text.remove_suffix(1);
    f.text = text;
    return f;
  }

 private:
  std::string_view data_;
  size_t pos_;
  char delim_;
};

// ---------------------------------------------------------------------------
// Value classification

std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return s;
}

bool parse_i64(std::string_view s, int64_t* out) {
  s = trim(s);
  if (s.empty()) return false;
  auto r = std::from_chars(s.data(), s.data() + s.size(), *out);
  return r.ec == std::errc() && r.ptr == s.data() + s.size();
}

bool parse_f64(std::string_view s, double* out) {
  s = trim(s);
  if (s.empty()) return false;
  auto r = std::from_chars(s.data(), s.data() + s.size(), *out);
  return r.ec == std::errc() && r.ptr == s.data() + s.size();
}

Type classify(std::string_view s) {
  int64_t i;
  double d;
  if (parse_i64(s, &i)) return Type::Int64;
  if (parse_f64(s, &d)) return Type::Float64;
  return Type::String;
}

inline Type widen(Type a, Type b) { return static_cast<uint8_t>(a) >= static_cast<uint8_t>(b) ? a : b; }

// ---------------------------------------------------------------------------
// Per-chunk column buffers

struct ColBuf {
  std::vector<int64_t> i64;
  std::vector<double> f64;
  std::vector<uint32_t> codes;
  std::vector<uint8_t> valid;
  bool any_null = false;
  std::unordered_map<std::string_view, uint32_t> dict_map;
  std::vector<std::string_view> dict;
  Type needed = Type::Int64;  // widest type observed that did not fit
};

struct Chunk {
  size_t begin = 0, end = 0;
  size_t rows = 0;
  std::vector<ColBuf> cols;
  std::deque<std::string> owned;  // unescaped quoted strings
};

void append(ColBuf& c, Type t, const Field& f) {
  const bool null = f.text.empty() && (!f.quoted || t != Type::String);
  c.valid.push_back(null ? 0x00 : 0xFF);
  if (null) c.any_null = true;
  switch (t) {
    case Type::Int64: {
      int64_t v = 0;
      if (!null && !parse_i64(f.text, &v)) c.needed = widen(c.needed, classify(f.text));
      c.i64.push_back(v);
      break;
    }
    case Type::Float64: {
      double v = 0;
      if (!null && !parse_f64(f.text, &v)) c.needed = widen(c.needed, Type::String);
      c.f64.push_back(v);
      break;
    }
    case Type::String: {
      uint32_t code = 0;
      if (!null) {
        auto [it, inserted] = c.dict_map.try_emplace(f.text, static_cast<uint32_t>(c.dict.size()));
        if (inserted) c.dict.push_back(f.text);
        code = it->second;
      }
      c.codes.push_back(code);
      break;
    }
  }
}

void parse_chunk(std::string_view data, char delim, const std::vector<Type>& types, Chunk& ch) {
  const size_t ncols = types.size();
  ch.cols.assign(ncols, ColBuf{});
  ch.owned.clear();
  ch.rows = 0;
  Reader rd(data.substr(0, ch.end), ch.begin, delim);
  while (rd.start_row()) {
    size_t col = 0;
    for (;;) {
      Field f = rd.next(&ch.owned);
      if (col >= ncols) throw std::runtime_error("csv: row has more fields than the header");
      append(ch.cols[col], types[col], f);
      ++col;
      if (f.end_of_row) break;
    }
    for (; col < ncols; ++col) append(ch.cols[col], types[col], Field{});
    ++ch.rows;
  }
}

// Splits [begin, size) into chunks that end just after a record-terminating
// newline. If the body contains quotes, a single sequential pass tracks the
// quote state so a newline inside a quoted field is never chosen.
std::vector<Chunk> split(std::string_view data, size_t begin, size_t target_chunks) {
  std::vector<Chunk> chunks;
  const size_t n = data.size();
  if (begin >= n) return chunks;
  const size_t body = n - begin;
  target_chunks = std::max<size_t>(1, target_chunks);
  const bool has_quotes = std::memchr(data.data() + begin, '"', body) != nullptr;

  size_t cur = begin;
  size_t scan = begin;
  bool in_quote = false;
  for (size_t k = 1; k < target_chunks && cur < n; ++k) {
    size_t target = begin + body / target_chunks * k;
    if (target <= cur) continue;
    size_t cut = n;
    if (!has_quotes) {
      const void* p = std::memchr(data.data() + target, '\n', n - target);
      if (p != nullptr) cut = static_cast<size_t>(static_cast<const char*>(p) - data.data()) + 1;
    } else {
      for (; scan < n; ++scan) {
        char c = data[scan];
        if (c == '"') {
          in_quote = !in_quote;
        } else if (c == '\n' && !in_quote && scan >= target) {
          cut = ++scan;
          break;
        }
      }
    }
    if (cut >= n) break;
    chunks.push_back(Chunk{cur, cut, 0, {}, {}});
    cur = cut;
  }
  if (cur < n) chunks.push_back(Chunk{cur, n, 0, {}, {}});
  return chunks;
}

}  // namespace

Table parse_csv(std::string_view data, const CsvOptions& opts) {
  const int threads = opts.threads > 0 ? opts.threads : default_threads();
  const char delim = opts.delimiter;
  size_t pos = 0;
  if (data.size() >= 3 && std::memcmp(data.data(), "\xEF\xBB\xBF", 3) == 0) pos = 3;

  // Header.
  std::vector<std::string> names;
  {
    Reader rd(data, pos, delim);
    if (!rd.start_row()) return Table{};
    std::deque<std::string> owned;
    std::vector<std::string> first;
    for (;;) {
      Field f = rd.next(&owned);
      first.emplace_back(f.text);
      if (f.end_of_row) break;
    }
    if (opts.header) {
      names = std::move(first);
      pos = rd.pos();
    } else {
      names.resize(first.size());
    }
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i].empty()) names[i] = "c" + std::to_string(i + 1);
    }
  }
  const size_t ncols = names.size();

  // Type inference from a sample.
  std::vector<Type> types(ncols, Type::Int64);
  {
    Reader rd(data, pos, delim);
    std::deque<std::string> owned;
    for (size_t r = 0; r < opts.sample_rows && rd.start_row(); ++r) {
      size_t col = 0;
      for (;;) {
        Field f = rd.next(&owned);
        if (col < ncols && !f.text.empty()) {
          types[col] = widen(types[col], classify(f.text));
        } else if (col < ncols && f.quoted) {
          types[col] = Type::String;  // "" is an empty string, not NULL
        }
        ++col;
        if (f.end_of_row) break;
      }
    }
  }

  std::vector<Chunk> chunks = split(data, pos,
                                    std::min<size_t>(static_cast<size_t>(threads) * 8,
                                                     std::max<size_t>(1, (data.size() - pos) / opts.min_chunk_bytes)));

  // Parse; promote any column that saw a value wider than its type and retry.
  for (int attempt = 0;; ++attempt) {
    parallel_for(chunks.size(), threads, [&](size_t i, int) { parse_chunk(data, delim, types, chunks[i]); });
    bool changed = false;
    for (size_t c = 0; c < ncols; ++c) {
      Type need = types[c];
      for (const auto& ch : chunks) need = widen(need, ch.cols[c].needed);
      if (need != types[c]) {
        types[c] = need;
        changed = true;
      }
    }
    if (!changed) break;
    if (attempt >= 2) throw std::runtime_error("csv: type promotion did not converge");
  }

  // Merge chunk buffers into contiguous columns.
  Table t;
  std::vector<size_t> offsets(chunks.size() + 1, 0);
  for (size_t i = 0; i < chunks.size(); ++i) offsets[i + 1] = offsets[i] + chunks[i].rows;
  t.num_rows = offsets.back();
  t.columns.resize(ncols);

  for (size_t c = 0; c < ncols; ++c) {
    Column& col = t.columns[c];
    col.name = names[c];
    col.type = types[c];
    bool any_null = false;
    for (const auto& ch : chunks) any_null |= ch.cols[c].any_null;
    if (any_null) col.valid.resize(t.num_rows);

    std::vector<std::vector<uint32_t>> remap;
    if (col.type == Type::String) {
      // Global dictionary in first-appearance order.
      std::unordered_map<std::string_view, uint32_t> global;
      remap.resize(chunks.size());
      for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& local = chunks[i].cols[c].dict;
        remap[i].resize(local.size());
        for (size_t k = 0; k < local.size(); ++k) {
          auto [it, inserted] = global.try_emplace(local[k], static_cast<uint32_t>(col.dict.size()));
          if (inserted) col.dict.emplace_back(local[k]);
          remap[i][k] = it->second;
        }
      }
      col.codes.resize(t.num_rows);
    } else if (col.type == Type::Int64) {
      col.i64.resize(t.num_rows);
    } else {
      col.f64.resize(t.num_rows);
    }

    parallel_for(chunks.size(), threads, [&](size_t i, int) {
      const ColBuf& b = chunks[i].cols[c];
      const size_t off = offsets[i];
      if (any_null) std::copy(b.valid.begin(), b.valid.end(), col.valid.begin() + static_cast<ptrdiff_t>(off));
      switch (col.type) {
        case Type::Int64: std::copy(b.i64.begin(), b.i64.end(), col.i64.begin() + static_cast<ptrdiff_t>(off)); break;
        case Type::Float64: std::copy(b.f64.begin(), b.f64.end(), col.f64.begin() + static_cast<ptrdiff_t>(off)); break;
        case Type::String: {
          const auto& rm = remap[i];
          for (size_t k = 0; k < b.codes.size(); ++k) {
            // NULL rows carry code 0 locally; remap only real values.
            col.codes[off + k] = b.valid[k] ? rm[b.codes[k]] : 0;
          }
          break;
        }
      }
    });
  }
  return t;
}

Table load_csv(const std::string& path, const CsvOptions& opts) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("cannot stat " + path);
  }
  const auto size = static_cast<size_t>(st.st_size);
  if (size == 0) {
    ::close(fd);
    return Table{};
  }
  void* p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("cannot mmap " + path);
  ::madvise(p, size, MADV_SEQUENTIAL);
  try {
    Table t = parse_csv(std::string_view(static_cast<const char*>(p), size), opts);
    ::munmap(p, size);
    return t;
  } catch (...) {
    ::munmap(p, size);
    throw;
  }
}

}  // namespace vecq
