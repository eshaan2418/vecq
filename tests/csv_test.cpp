#include "vecq/csv.h"

#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
#include <string>

using vecq::CsvOptions;
using vecq::parse_csv;
using vecq::Type;

TEST(Csv, InfersTypes) {
  auto t = parse_csv("a,b,c\n1,2.5,x\n-3,4,y\n");
  ASSERT_EQ(t.num_rows, 2u);
  ASSERT_EQ(t.columns.size(), 3u);
  EXPECT_EQ(t.columns[0].type, Type::Int64);
  EXPECT_EQ(t.columns[1].type, Type::Float64);
  EXPECT_EQ(t.columns[2].type, Type::String);
  EXPECT_EQ(t.columns[0].i64[1], -3);
  EXPECT_DOUBLE_EQ(t.columns[1].f64[1], 4.0);
  EXPECT_EQ(t.columns[2].dict[t.columns[2].codes[1]], "y");
  EXPECT_FALSE(t.columns[0].has_nulls());
}

TEST(Csv, QuotesEscapesAndEmbeddedNewlines) {
  auto t = parse_csv("id,note\n1,\"hello, world\"\n2,\"say \"\"hi\"\"\"\n3,\"line1\nline2\"\n");
  ASSERT_EQ(t.num_rows, 3u);
  const auto& c = t.columns[1];
  EXPECT_EQ(c.dict[c.codes[0]], "hello, world");
  EXPECT_EQ(c.dict[c.codes[1]], "say \"hi\"");
  EXPECT_EQ(c.dict[c.codes[2]], "line1\nline2");
}

TEST(Csv, CrlfBomAndBlankLines) {
  auto t = parse_csv("\xEF\xBB\xBFx,y\r\n1,a\r\n\r\n2,b\r\n");
  ASSERT_EQ(t.num_rows, 2u);
  EXPECT_EQ(t.columns[0].name, "x");
  EXPECT_EQ(t.columns[1].dict[t.columns[1].codes[1]], "b");
}

TEST(Csv, NullsAndShortRows) {
  auto t = parse_csv("a,b,s\n1,,x\n,2.5,\n3\n");
  ASSERT_EQ(t.num_rows, 3u);
  const auto& a = t.columns[0];
  const auto& b = t.columns[1];
  const auto& s = t.columns[2];
  EXPECT_TRUE(a.has_nulls());
  EXPECT_TRUE(a.is_valid(0));
  EXPECT_FALSE(a.is_valid(1));
  EXPECT_FALSE(b.is_valid(0));
  EXPECT_FALSE(b.is_valid(2));
  EXPECT_FALSE(s.is_valid(1));
  EXPECT_FALSE(s.is_valid(2));
}

TEST(Csv, QuotedEmptyIsEmptyString) {
  auto t = parse_csv("s\n\"\"\nx\n");
  const auto& s = t.columns[0];
  ASSERT_EQ(s.type, Type::String);
  EXPECT_TRUE(s.is_valid(0));
  EXPECT_EQ(s.dict[s.codes[0]], "");
}

TEST(Csv, PromotesWhenLateValueDoesNotFit) {
  CsvOptions o;
  o.sample_rows = 2;
  auto t = parse_csv("a,b\n1,1\n2,2\n3.5,x\n", o);
  EXPECT_EQ(t.columns[0].type, Type::Float64);
  EXPECT_EQ(t.columns[1].type, Type::String);
  EXPECT_DOUBLE_EQ(t.columns[0].f64[2], 3.5);
  EXPECT_EQ(t.columns[1].dict[t.columns[1].codes[0]], "1");
}

TEST(Csv, NoHeader) {
  CsvOptions o;
  o.header = false;
  auto t = parse_csv("1,2\n3,4\n", o);
  ASSERT_EQ(t.num_rows, 2u);
  EXPECT_EQ(t.columns[0].name, "c1");
  EXPECT_EQ(t.columns[1].i64[1], 4);
}

TEST(Csv, CustomDelimiter) {
  CsvOptions o;
  o.delimiter = '\t';
  auto t = parse_csv("a\tb\n1\thello world\n", o);
  EXPECT_EQ(t.columns[1].dict[0], "hello world");
}

TEST(Csv, Errors) {
  EXPECT_THROW(parse_csv("a\n\"open\n"), std::runtime_error);
  EXPECT_THROW(parse_csv("a,b\n1,2,3\n"), std::runtime_error);
  EXPECT_THROW(parse_csv("a\n\"x\"y\n"), std::runtime_error);
  EXPECT_THROW(vecq::load_csv("/nonexistent/file.csv"), std::runtime_error);
}

TEST(Csv, EmptyInput) {
  EXPECT_EQ(parse_csv("").num_rows, 0u);
  auto t = parse_csv("a,b\n");
  EXPECT_EQ(t.num_rows, 0u);
  EXPECT_EQ(t.columns.size(), 2u);
}

// Parallel chunked parsing (including chunk boundaries next to quoted
// newlines) must produce exactly the same table as a single-threaded parse.
TEST(Csv, ParallelMatchesSerial) {
  std::mt19937_64 rng(9);
  std::string data = "id,name,val\n";
  for (int i = 0; i < 20000; ++i) {
    data += std::to_string(i) + ",";
    switch (rng() % 4) {
      case 0: data += "\"multi\nline " + std::to_string(i % 37) + "\""; break;
      case 1: data += "\"q\"\"" + std::to_string(i % 11) + "\""; break;
      case 2: break;  // NULL
      default: data += "plain" + std::to_string(i % 13);
    }
    data += "," + std::to_string(static_cast<double>(rng() % 1000) / 8.0) + "\n";
  }
  CsvOptions serial;
  serial.threads = 1;
  CsvOptions par;
  par.threads = 8;
  par.min_chunk_bytes = 512;
  auto a = parse_csv(data, serial);
  auto b = parse_csv(data, par);
  ASSERT_EQ(a.num_rows, 20000u);
  ASSERT_EQ(a.num_rows, b.num_rows);
  for (size_t c = 0; c < a.columns.size(); ++c) {
    const auto& x = a.columns[c];
    const auto& y = b.columns[c];
    ASSERT_EQ(x.type, y.type);
    ASSERT_EQ(x.valid, y.valid);
    ASSERT_EQ(x.i64, y.i64);
    ASSERT_EQ(x.f64, y.f64);
    ASSERT_EQ(x.codes.size(), y.codes.size());
    for (size_t r = 0; r < x.codes.size(); ++r) {
      if (!x.is_valid(r)) continue;
      ASSERT_EQ(x.dict[x.codes[r]], y.dict[y.codes[r]]) << "row " << r;
    }
  }
}
