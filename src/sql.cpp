#include "vecq/sql.h"

#include <cctype>
#include <charconv>
#include <stdexcept>

#include "vecq/table.h"

namespace vecq::sql {

const char* agg_name(AggFunc f) {
  switch (f) {
    case AggFunc::Count: return "COUNT";
    case AggFunc::Sum: return "SUM";
    case AggFunc::Avg: return "AVG";
    case AggFunc::Min: return "MIN";
    case AggFunc::Max: return "MAX";
  }
  return "?";
}

namespace {

struct Token {
  enum class Kind : uint8_t { Ident, Int, Float, String, Symbol, End } kind = Kind::End;
  std::string text;
  size_t pos = 0;
};

std::vector<Token> lex(const std::string& q) {
  std::vector<Token> out;
  size_t i = 0;
  const size_t n = q.size();
  auto err = [&](const std::string& m) { throw std::runtime_error("sql: " + m + " at offset " + std::to_string(i)); };
  while (i < n) {
    char c = q[i];
    if (std::isspace(static_cast<unsigned char>(c))) {
      ++i;
      continue;
    }
    Token t;
    t.pos = i;
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t s = i;
      while (i < n && (std::isalnum(static_cast<unsigned char>(q[i])) || q[i] == '_' || q[i] == '.')) ++i;
      t.kind = Token::Kind::Ident;
      t.text = q.substr(s, i - s);
    } else if (c == '"' || c == '`') {
      // Quoted identifier.
      char close = c;
      size_t s = ++i;
      while (i < n && q[i] != close) ++i;
      if (i >= n) err("unterminated quoted identifier");
      t.kind = Token::Kind::Ident;
      t.text = q.substr(s, i - s);
      ++i;
    } else if (std::isdigit(static_cast<unsigned char>(c)) ||
               (c == '.' && i + 1 < n && std::isdigit(static_cast<unsigned char>(q[i + 1])))) {
      size_t s = i;
      bool is_float = false;
      while (i < n && std::isdigit(static_cast<unsigned char>(q[i]))) ++i;
      if (i < n && q[i] == '.') {
        is_float = true;
        ++i;
        while (i < n && std::isdigit(static_cast<unsigned char>(q[i]))) ++i;
      }
      if (i < n && (q[i] == 'e' || q[i] == 'E')) {
        is_float = true;
        ++i;
        if (i < n && (q[i] == '+' || q[i] == '-')) ++i;
        if (i >= n || !std::isdigit(static_cast<unsigned char>(q[i]))) err("bad exponent");
        while (i < n && std::isdigit(static_cast<unsigned char>(q[i]))) ++i;
      }
      t.kind = is_float ? Token::Kind::Float : Token::Kind::Int;
      t.text = q.substr(s, i - s);
    } else if (c == '\'') {
      ++i;
      std::string s;
      for (;;) {
        if (i >= n) err("unterminated string literal");
        if (q[i] == '\'') {
          if (i + 1 < n && q[i + 1] == '\'') {
            s.push_back('\'');
            i += 2;
            continue;
          }
          ++i;
          break;
        }
        s.push_back(q[i++]);
      }
      t.kind = Token::Kind::String;
      t.text = std::move(s);
    } else {
      static const char* two[] = {"<=", ">=", "!=", "<>"};
      t.kind = Token::Kind::Symbol;
      bool matched = false;
      for (const char* s : two) {
        if (q.compare(i, 2, s) == 0) {
          t.text = s;
          i += 2;
          matched = true;
          break;
        }
      }
      if (!matched) {
        if (std::string("(),*=<>-;").find(c) == std::string::npos) err(std::string("unexpected character '") + c + "'");
        t.text = std::string(1, c);
        ++i;
      }
    }
    out.push_back(std::move(t));
  }
  Token end;
  end.pos = n;
  out.push_back(end);
  return out;
}

class Parser {
 public:
  explicit Parser(std::vector<Token> toks) : t_(std::move(toks)) {}

  Select parse_select() {
    Select s;
    expect_kw("SELECT");
    do {
      s.items.push_back(parse_item());
    } while (accept_sym(","));
    expect_kw("FROM");
    s.table = expect_ident("table name");
    if (accept_kw("WHERE")) s.where = parse_or();
    if (accept_kw("GROUP")) {
      expect_kw("BY");
      do {
        s.group_by.push_back(expect_ident("column"));
      } while (accept_sym(","));
    }
    if (accept_kw("ORDER")) {
      expect_kw("BY");
      do {
        OrderItem o;
        if (peek().kind == Token::Kind::Int) {
          o.ordinal = std::stoi(next().text);
          if (*o.ordinal < 1) fail("ORDER BY ordinal must be >= 1");
        } else {
          o.key = parse_ref(/*allow_alias=*/true);
        }
        if (accept_kw("DESC")) {
          o.desc = true;
        } else {
          accept_kw("ASC");
        }
        s.order_by.push_back(std::move(o));
      } while (accept_sym(","));
    }
    if (accept_kw("LIMIT")) {
      if (peek().kind != Token::Kind::Int) fail("LIMIT expects an integer");
      s.limit = std::stoll(next().text);
    }
    accept_sym(";");
    if (peek().kind != Token::Kind::End) fail("unexpected token '" + peek().text + "'");
    return s;
  }

 private:
  const Token& peek(size_t k = 0) const { return t_[std::min(p_ + k, t_.size() - 1)]; }
  Token next() { return t_[std::min(p_++, t_.size() - 1)]; }

  [[noreturn]] void fail(const std::string& m) const {
    throw std::runtime_error("sql: " + m + " at offset " + std::to_string(peek().pos));
  }

  bool is_kw(const Token& t, const char* kw) const { return t.kind == Token::Kind::Ident && iequals(t.text, kw); }
  bool accept_kw(const char* kw) {
    if (is_kw(peek(), kw)) {
      ++p_;
      return true;
    }
    return false;
  }
  void expect_kw(const char* kw) {
    if (!accept_kw(kw)) fail(std::string("expected ") + kw);
  }
  bool accept_sym(const char* s) {
    if (peek().kind == Token::Kind::Symbol && peek().text == s) {
      ++p_;
      return true;
    }
    return false;
  }
  void expect_sym(const char* s) {
    if (!accept_sym(s)) fail(std::string("expected '") + s + "'");
  }
  std::string expect_ident(const char* what) {
    if (peek().kind != Token::Kind::Ident) fail(std::string("expected ") + what);
    return next().text;
  }

  static std::optional<AggFunc> agg_of(const std::string& s) {
    if (iequals(s, "COUNT")) return AggFunc::Count;
    if (iequals(s, "SUM")) return AggFunc::Sum;
    if (iequals(s, "AVG")) return AggFunc::Avg;
    if (iequals(s, "MIN")) return AggFunc::Min;
    if (iequals(s, "MAX")) return AggFunc::Max;
    return std::nullopt;
  }

  // Column or aggregate reference.
  SelectItem parse_ref(bool allow_alias) {
    (void)allow_alias;
    SelectItem it;
    std::string name = expect_ident("column or aggregate");
    if (auto f = agg_of(name); f && accept_sym("(")) {
      it.kind = SelectItem::Kind::Agg;
      it.func = *f;
      if (accept_sym("*")) {
        if (*f != AggFunc::Count) fail("only COUNT accepts *");
        it.text = std::string(agg_name(*f)) + "(*)";
      } else {
        it.column = expect_ident("column");
        it.text = std::string(agg_name(*f)) + "(" + it.column + ")";
      }
      expect_sym(")");
      return it;
    }
    it.kind = SelectItem::Kind::Column;
    it.column = name;
    it.text = name;
    return it;
  }

  SelectItem parse_item() {
    if (accept_sym("*")) {
      SelectItem it;
      it.kind = SelectItem::Kind::Star;
      it.text = "*";
      return it;
    }
    SelectItem it = parse_ref(false);
    if (accept_kw("AS")) it.alias = expect_ident("alias");
    return it;
  }

  Literal parse_literal() {
    bool neg = accept_sym("-");
    Token t = next();
    Literal l;
    switch (t.kind) {
      case Token::Kind::Int: {
        l.kind = Literal::Kind::Int;
        auto r = std::from_chars(t.text.data(), t.text.data() + t.text.size(), l.i);
        if (r.ec != std::errc()) {
          // Out of int64 range: fall back to double.
          l.kind = Literal::Kind::Float;
          l.f = std::stod(t.text);
          if (neg) l.f = -l.f;
        } else if (neg) {
          l.i = -l.i;
        }
        break;
      }
      case Token::Kind::Float:
        l.kind = Literal::Kind::Float;
        l.f = std::stod(t.text);
        if (neg) l.f = -l.f;
        break;
      case Token::Kind::String:
        if (neg) fail("cannot negate a string");
        l.kind = Literal::Kind::String;
        l.s = t.text;
        break;
      default: --p_; fail("expected a literal");
    }
    return l;
  }

  bool peek_literal() const {
    auto k = peek().kind;
    return k == Token::Kind::Int || k == Token::Kind::Float || k == Token::Kind::String ||
           (peek().kind == Token::Kind::Symbol && peek().text == "-");
  }

  std::optional<kernels::CmpOp> accept_cmp() {
    if (peek().kind != Token::Kind::Symbol) return std::nullopt;
    const std::string& s = peek().text;
    std::optional<kernels::CmpOp> op;
    if (s == "=")
      op = kernels::CmpOp::Eq;
    else if (s == "!=" || s == "<>")
      op = kernels::CmpOp::Ne;
    else if (s == "<")
      op = kernels::CmpOp::Lt;
    else if (s == "<=")
      op = kernels::CmpOp::Le;
    else if (s == ">")
      op = kernels::CmpOp::Gt;
    else if (s == ">=")
      op = kernels::CmpOp::Ge;
    if (op) ++p_;
    return op;
  }

  static kernels::CmpOp flip(kernels::CmpOp op) {
    using kernels::CmpOp;
    switch (op) {
      case CmpOp::Lt: return CmpOp::Gt;
      case CmpOp::Le: return CmpOp::Ge;
      case CmpOp::Gt: return CmpOp::Lt;
      case CmpOp::Ge: return CmpOp::Le;
      default: return op;
    }
  }

  static std::unique_ptr<Pred> binary(Pred::Kind k, std::unique_ptr<Pred> l, std::unique_ptr<Pred> r) {
    auto p = std::make_unique<Pred>();
    p->kind = k;
    p->lhs = std::move(l);
    p->rhs = std::move(r);
    return p;
  }

  static std::unique_ptr<Pred> cmp(std::string col, kernels::CmpOp op, Literal lit) {
    auto p = std::make_unique<Pred>();
    p->kind = Pred::Kind::Cmp;
    p->column = std::move(col);
    p->op = op;
    p->lit = std::move(lit);
    return p;
  }

  std::unique_ptr<Pred> parse_or() {
    auto l = parse_and();
    while (accept_kw("OR")) l = binary(Pred::Kind::Or, std::move(l), parse_and());
    return l;
  }

  std::unique_ptr<Pred> parse_and() {
    auto l = parse_atom();
    while (accept_kw("AND")) l = binary(Pred::Kind::And, std::move(l), parse_atom());
    return l;
  }

  std::unique_ptr<Pred> parse_atom() {
    if (accept_sym("(")) {
      auto p = parse_or();
      expect_sym(")");
      return p;
    }
    if (is_kw(peek(), "NOT")) fail("NOT is only supported as NOT LIKE / NOT BETWEEN / IS NOT NULL");
    if (peek_literal()) {
      Literal lit = parse_literal();
      auto op = accept_cmp();
      if (!op) fail("expected comparison operator");
      return cmp(expect_ident("column"), flip(*op), std::move(lit));
    }
    std::string col = expect_ident("column");
    if (auto op = accept_cmp()) return cmp(col, *op, parse_literal());
    if (accept_kw("IS")) {
      auto p = std::make_unique<Pred>();
      p->kind = Pred::Kind::IsNull;
      p->column = col;
      p->negated = accept_kw("NOT");
      expect_kw("NULL");
      return p;
    }
    bool neg = accept_kw("NOT");
    if (accept_kw("LIKE")) {
      if (peek().kind != Token::Kind::String) fail("LIKE expects a string pattern");
      auto p = std::make_unique<Pred>();
      p->kind = Pred::Kind::Like;
      p->column = col;
      p->negated = neg;
      p->lit.kind = Literal::Kind::String;
      p->lit.s = next().text;
      return p;
    }
    if (accept_kw("BETWEEN")) {
      Literal lo = parse_literal();
      expect_kw("AND");
      Literal hi = parse_literal();
      if (!neg) {
        return binary(Pred::Kind::And, cmp(col, kernels::CmpOp::Ge, std::move(lo)),
                      cmp(col, kernels::CmpOp::Le, std::move(hi)));
      }
      return binary(Pred::Kind::Or, cmp(col, kernels::CmpOp::Lt, std::move(lo)),
                    cmp(col, kernels::CmpOp::Gt, std::move(hi)));
    }
    fail("expected comparison, LIKE, BETWEEN or IS after column");
  }

  std::vector<Token> t_;
  size_t p_ = 0;
};

}  // namespace

Select parse(const std::string& query) { return Parser(lex(query)).parse_select(); }

}  // namespace vecq::sql
