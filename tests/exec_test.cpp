#include "vecq/exec.h"

#include <gtest/gtest.h>

#include <stdexcept>

#include "vecq/csv.h"

using namespace vecq;

namespace {

const char* kData =
    "id,city,temp,rain\n"
    "1,oslo,4,2.5\n"
    "2,lima,19,\n"
    "3,oslo,-2,7.0\n"
    "4,cairo,31,0\n"
    "5,lima,21,0.5\n"
    "6,,15,1.0\n";

Table table() { return parse_csv(kData); }

int64_t i(const Value& v) { return std::get<int64_t>(v); }
double d(const Value& v) { return std::get<double>(v); }
const std::string& s(const Value& v) { return std::get<std::string>(v); }
bool null(const Value& v) { return v.index() == 0; }

}  // namespace

TEST(Exec, CountAndFilter) {
  auto t = table();
  auto r = query(t, "SELECT COUNT(*), COUNT(rain), COUNT(city) FROM t");
  ASSERT_EQ(r.rows.size(), 1u);
  EXPECT_EQ(i(r.rows[0][0]), 6);
  EXPECT_EQ(i(r.rows[0][1]), 5);
  EXPECT_EQ(i(r.rows[0][2]), 5);
  r = query(t, "SELECT COUNT(*) FROM t WHERE temp > 10 AND rain < 1");
  EXPECT_EQ(i(r.rows[0][0]), 2);  // cairo, lima(21)
}

TEST(Exec, SumAvgMinMaxWithNulls) {
  auto t = table();
  auto r = query(t, "SELECT SUM(temp), AVG(rain), MIN(rain), MAX(temp) FROM t");
  EXPECT_EQ(i(r.rows[0][0]), 88);
  EXPECT_DOUBLE_EQ(d(r.rows[0][1]), (2.5 + 7.0 + 0 + 0.5 + 1.0) / 5);
  EXPECT_DOUBLE_EQ(d(r.rows[0][2]), 0.0);
  EXPECT_EQ(i(r.rows[0][3]), 31);
}

TEST(Exec, EmptyAggregatesAreNull) {
  auto t = table();
  auto r = query(t, "SELECT COUNT(*), SUM(temp), AVG(temp), MIN(rain) FROM t WHERE temp > 100");
  EXPECT_EQ(i(r.rows[0][0]), 0);
  EXPECT_TRUE(null(r.rows[0][1]));
  EXPECT_TRUE(null(r.rows[0][2]));
  EXPECT_TRUE(null(r.rows[0][3]));
}

TEST(Exec, GroupByWithNullKeyAndOrder) {
  auto t = table();
  auto r = query(t, "SELECT city, COUNT(*) AS n, SUM(temp) FROM t GROUP BY city ORDER BY n DESC, city");
  ASSERT_EQ(r.rows.size(), 4u);
  EXPECT_EQ(r.names[1], "n");
  EXPECT_EQ(s(r.rows[0][0]), "lima");
  EXPECT_EQ(i(r.rows[0][2]), 40);
  EXPECT_EQ(s(r.rows[1][0]), "oslo");
  EXPECT_TRUE(null(r.rows[2][0]));  // NULL sorts first among n=1
  EXPECT_EQ(s(r.rows[3][0]), "cairo");
}

TEST(Exec, MultiKeyGroupBy) {
  auto t = parse_csv("a,b,v\n1,x,1\n1,y,2\n1,x,3\n2,x,4\n");
  auto r = query(t, "SELECT a, b, SUM(v) FROM t GROUP BY a, b ORDER BY a, b");
  ASSERT_EQ(r.rows.size(), 3u);
  EXPECT_EQ(i(r.rows[0][2]), 4);
  EXPECT_EQ(s(r.rows[1][1]), "y");
  EXPECT_EQ(i(r.rows[2][2]), 4);
}

TEST(Exec, ProjectionOrderLimit) {
  auto t = table();
  auto r = query(t, "SELECT id, city FROM t WHERE rain IS NOT NULL ORDER BY id DESC LIMIT 2");
  ASSERT_EQ(r.rows.size(), 2u);
  EXPECT_EQ(i(r.rows[0][0]), 6);
  EXPECT_EQ(i(r.rows[1][0]), 5);
  r = query(t, "SELECT * FROM t LIMIT 3");
  ASSERT_EQ(r.rows.size(), 3u);
  EXPECT_EQ(r.names.size(), 4u);
}

TEST(Exec, StringPredicates) {
  auto t = table();
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE city = 'oslo'").rows[0][0]), 2);
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE city != 'oslo'").rows[0][0]), 3);  // NULL excluded
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE city = 'paris'").rows[0][0]), 0);
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE city LIKE '%i%'").rows[0][0]), 3);
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE city NOT LIKE '_slo'").rows[0][0]), 3);
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE city < 'm'").rows[0][0]), 3);
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE city IS NULL").rows[0][0]), 1);
}

TEST(Exec, IntColumnVsFractionalLiteral) {
  auto t = table();
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE temp > 18.5").rows[0][0]), 3);
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE temp <= 4.9").rows[0][0]), 2);
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE temp = 4.5").rows[0][0]), 0);
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE temp != 4.5").rows[0][0]), 6);
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE temp < 1e30").rows[0][0]), 6);
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE temp BETWEEN 4 AND 19").rows[0][0]), 3);
  EXPECT_EQ(i(query(t, "SELECT COUNT(*) FROM t WHERE temp NOT BETWEEN 4 AND 19").rows[0][0]), 3);
}

TEST(Exec, SemanticErrors) {
  auto t = table();
  EXPECT_THROW(query(t, "SELECT nope FROM t"), std::runtime_error);
  EXPECT_THROW(query(t, "SELECT city, COUNT(*) FROM t"), std::runtime_error);
  EXPECT_THROW(query(t, "SELECT SUM(city) FROM t"), std::runtime_error);
  EXPECT_THROW(query(t, "SELECT * FROM t WHERE temp = 'x'"), std::runtime_error);
  EXPECT_THROW(query(t, "SELECT * FROM t WHERE city > 3"), std::runtime_error);
  EXPECT_THROW(query(t, "SELECT * FROM t WHERE temp LIKE 'x'"), std::runtime_error);
  EXPECT_THROW(query(t, "SELECT id FROM t ORDER BY temp"), std::runtime_error);
  EXPECT_THROW(query(t, "SELECT id FROM t ORDER BY 2"), std::runtime_error);
}

TEST(Exec, Formatting) {
  auto t = table();
  auto r = query(t, "SELECT city, MAX(rain) FROM t WHERE city = 'oslo' GROUP BY city");
  EXPECT_EQ(format_csv(r), "city,MAX(rain)\noslo,7\n");
  std::string tbl = format_table(r);
  EXPECT_NE(tbl.find("| oslo |"), std::string::npos);
  EXPECT_NE(tbl.find("1 row"), std::string::npos);
}
