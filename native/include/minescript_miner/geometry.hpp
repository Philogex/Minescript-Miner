#pragma once

#include "minescript_miner/tri2.hpp"

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/rational.hpp>

#include <cstdint>

namespace minescript_miner {

using ExactInt = boost::multiprecision::cpp_int;
using ExactRational = boost::rational<ExactInt>;

enum class ExactSign : std::int8_t {
    Negative = -1,
    Zero = 0,
    Positive = 1,
};

struct ExactPoint2H {
    ExactInt x = 0;
    ExactInt y = 0;
    ExactInt w = 1;

    bool operator==(const ExactPoint2H &other) const {
        return x == other.x && y == other.y && w == other.w;
    }
};

struct ExactLine2 {
    ExactInt a = 0;
    ExactInt b = 0;
    ExactInt c = 0;

    bool operator==(const ExactLine2 &other) const {
        return a == other.a && b == other.b && c == other.c;
    }
};

ExactRational rational_from_double(double value);
double approximate_double(const ExactRational &value);

ExactPoint2H normalize_point(ExactPoint2H point);
ExactLine2 normalize_line(ExactLine2 line);

ExactPoint2H make_point(const ExactRational &x, const ExactRational &y);
ExactPoint2H make_point(double x, double y);
Point2 approximate_point(const ExactPoint2H &point);

ExactLine2 line_through(
    const ExactPoint2H &from,
    const ExactPoint2H &to
);
ExactPoint2H line_intersection(
    const ExactLine2 &lhs,
    const ExactLine2 &rhs
);

ExactLine2 opposite_half_plane(ExactLine2 line);
ExactSign classify_line(
    const ExactLine2 &line,
    const ExactPoint2H &point
);

bool is_finite(const ExactPoint2H &point);
bool is_valid(const ExactPoint2H &point);
bool is_valid(const ExactLine2 &line);

}  // namespace minescript_miner
