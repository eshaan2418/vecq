#include "vecq/sql.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace vecq::sql;
using vecq::kernels::CmpOp;

TEST(Sql, ParsesFullQuery) {
  auto s = parse(
      "select payment, count(*) as n, avg(fare) from trips where fare > 2.5 and (zone = 'Z001' or tip >= 1) "
      "group by payment order by n desc, 1 limit 10;");
  ASSERT_EQ(s.items.size(), 3u);
  EXPECT_EQ(s.items[0].kind, SelectItem::Kind::Column);
  EXPECT_EQ(s.items[1].kind, SelectItem::Kind::Agg);
  EXPECT_EQ(s.items[1].alias, "n");
  EXPECT_EQ(s.items[1].text, "COUNT(*)");
  EXPECT_EQ(s.items[2].text, "AVG(fare)");
  EXPECT_EQ(s.table, "trips");
  ASSERT_TRUE(s.where);
  EXPECT_EQ(s.where->kind, Pred::Kind::And);
  EXPECT_EQ(s.where->rhs->kind, Pred::Kind::Or);
  ASSERT_EQ(s.group_by.size(), 1u);
  ASSERT_EQ(s.order_by.size(), 2u);
  EXPECT_TRUE(s.order_by[0].desc);
  EXPECT_EQ(*s.order_by[1].ordinal, 1);
  EXPECT_EQ(*s.limit, 10);
}

TEST(Sql, AndBindsTighterThanOr) {
  auto s = parse("SELECT * FROM t WHERE a = 1 OR b = 2 AND c = 3");
  EXPECT_EQ(s.where->kind, Pred::Kind::Or);
  EXPECT_EQ(s.where->rhs->kind, Pred::Kind::And);
}

TEST(Sql, LiteralOnLeftIsFlipped) {
  auto s = parse("SELECT * FROM t WHERE 5 < x");
  EXPECT_EQ(s.where->column, "x");
  EXPECT_EQ(s.where->op, CmpOp::Gt);
  EXPECT_EQ(s.where->lit.i, 5);
}

TEST(Sql, NegativeAndFloatLiterals) {
  auto s = parse("SELECT * FROM t WHERE x >= -2.5e1 AND y <> -7");
  EXPECT_EQ(s.where->lhs->lit.kind, Literal::Kind::Float);
  EXPECT_DOUBLE_EQ(s.where->lhs->lit.f, -25.0);
  EXPECT_EQ(s.where->rhs->op, CmpOp::Ne);
  EXPECT_EQ(s.where->rhs->lit.i, -7);
}

TEST(Sql, BetweenLikeIsNull) {
  auto s = parse("SELECT * FROM t WHERE x BETWEEN 1 AND 5 AND s NOT LIKE 'a%' AND y IS NOT NULL");
  // ((x BETWEEN) AND (s NOT LIKE)) AND (y IS NOT NULL)
  EXPECT_EQ(s.where->rhs->kind, Pred::Kind::IsNull);
  EXPECT_TRUE(s.where->rhs->negated);
  EXPECT_EQ(s.where->lhs->rhs->kind, Pred::Kind::Like);
  EXPECT_TRUE(s.where->lhs->rhs->negated);
  EXPECT_EQ(s.where->lhs->lhs->kind, Pred::Kind::And);
}

TEST(Sql, StringEscapesAndQuotedIdentifiers) {
  auto s = parse("SELECT \"my col\" FROM t WHERE name = 'O''Brien'");
  EXPECT_EQ(s.items[0].column, "my col");
  EXPECT_EQ(s.where->lit.s, "O'Brien");
}

TEST(Sql, Errors) {
  EXPECT_THROW(parse("SELECT FROM t"), std::runtime_error);
  EXPECT_THROW(parse("SELECT a FROM"), std::runtime_error);
  EXPECT_THROW(parse("SELECT a FROM t WHERE"), std::runtime_error);
  EXPECT_THROW(parse("SELECT a FROM t WHERE a = 'x"), std::runtime_error);
  EXPECT_THROW(parse("SELECT SUM(*) FROM t"), std::runtime_error);
  EXPECT_THROW(parse("SELECT a FROM t LIMIT x"), std::runtime_error);
  EXPECT_THROW(parse("SELECT a FROM t garbage"), std::runtime_error);
  EXPECT_THROW(parse("SELECT a FROM t WHERE NOT a = 1"), std::runtime_error);
  EXPECT_THROW(parse("SELECT a FROM t WHERE a # 1"), std::runtime_error);
}
