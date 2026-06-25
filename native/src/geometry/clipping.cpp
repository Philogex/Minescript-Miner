#include "minescript_miner/geometry/clipping.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace minescript_miner {

static_assert(
    std::numeric_limits<double>::is_iec559,
    "robust predicates require IEEE 754 double precision"
);
static_assert(orient2d_determinant({0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}) > 0.0);
static_assert(orient2d_determinant({0.0, 0.0}, {1.0, 0.0}, {0.0, -1.0}) < 0.0);
static_assert(orient2d_determinant({0.0, 0.0}, {1.0, 0.0}, {0.5, 0.0}) == 0.0);

namespace {

constexpr double SPLITTER = 134217729.0;
constexpr double UNIT_ROUNDOFF = std::numeric_limits<double>::epsilon() * 0.5;
constexpr double ORIENT_ERROR_BOUND_A =
    (3.0 + 16.0 * UNIT_ROUNDOFF) * UNIT_ROUNDOFF;

Orientation orientation_from_value(double value) {
    if (value > 0.0) {
        return Orientation::CounterClockwise;
    }
    if (value < 0.0) {
        return Orientation::Clockwise;
    }
    return Orientation::Collinear;
}

void two_sum(double a, double b, double &sum, double &error) {
    sum = a + b;
    const double b_virtual = sum - a;
    const double a_virtual = sum - b_virtual;
    const double b_roundoff = b - b_virtual;
    const double a_roundoff = a - a_virtual;
    error = a_roundoff + b_roundoff;
}

void split(double value, double &high, double &low) {
    const double combined = SPLITTER * value;
    const double large = combined - value;
    high = combined - large;
    low = value - high;
}

void two_product(double a, double b, double &product, double &error) {
    product = a * b;

    double a_high = 0.0;
    double a_low = 0.0;
    double b_high = 0.0;
    double b_low = 0.0;
    split(a, a_high, a_low);
    split(b, b_high, b_low);

    const double error1 = product - a_high * b_high;
    const double error2 = error1 - a_low * b_high;
    const double error3 = error2 - a_high * b_low;
    error = a_low * b_low - error3;
}

template <std::size_t Capacity>
std::size_t grow_expansion(
    const std::array<double, Capacity> &input,
    std::size_t input_length,
    double value,
    std::array<double, Capacity> &output
) {
    double accumulator = value;
    std::size_t output_length = 0;
    for (std::size_t i = 0; i < input_length; ++i) {
        double sum = 0.0;
        double error = 0.0;
        two_sum(accumulator, input[i], sum, error);
        if (error != 0.0) {
            output[output_length++] = error;
        }
        accumulator = sum;
    }
    if (accumulator != 0.0 || output_length == 0) {
        output[output_length++] = accumulator;
    }
    return output_length;
}

template <std::size_t Capacity>
void add_expansion_value(
    std::array<double, Capacity> &expansion,
    std::array<double, Capacity> &scratch,
    std::size_t &length,
    double value
) {
    length = grow_expansion(expansion, length, value, scratch);
    expansion = scratch;
    scratch = {};
}

template <std::size_t Capacity>
void add_expansion_product(
    std::array<double, Capacity> &expansion,
    std::array<double, Capacity> &scratch,
    std::size_t &length,
    double lhs,
    double rhs,
    double sign
) {
    double product = 0.0;
    double error = 0.0;
    two_product(lhs, rhs, product, error);
    add_expansion_value(expansion, scratch, length, sign * error);
    add_expansion_value(expansion, scratch, length, sign * product);
}

template <std::size_t Capacity>
double most_significant_component(
    const std::array<double, Capacity> &expansion,
    std::size_t length
) {
    for (std::size_t i = length; i > 0; --i) {
        if (expansion[i - 1] != 0.0) {
            return expansion[i - 1];
        }
    }
    return 0.0;
}

double expansion_orientation_value(Point2 a, Point2 b, Point2 point) {
    constexpr std::size_t MAX_COMPONENTS = 16;
    std::array<double, MAX_COMPONENTS> expansion{};
    std::array<double, MAX_COMPONENTS> scratch{};
    std::size_t length = 1;

    add_expansion_product(expansion, scratch, length, a.x, b.y, 1.0);
    add_expansion_product(expansion, scratch, length, a.y, b.x, -1.0);
    add_expansion_product(expansion, scratch, length, b.x, point.y, 1.0);
    add_expansion_product(expansion, scratch, length, b.y, point.x, -1.0);
    add_expansion_product(expansion, scratch, length, point.x, a.y, 1.0);
    add_expansion_product(expansion, scratch, length, point.y, a.x, -1.0);

    return most_significant_component(expansion, length);
}

// Fast error filter plus an exact expansion fallback, following the adaptive
// predicate strategy described by Jonathan Shewchuk.
double robust_orientation_value(Point2 a, Point2 b, Point2 point) {
    const double determinant_left =
        (a.x - point.x) * (b.y - point.y);
    const double determinant_right =
        (a.y - point.y) * (b.x - point.x);
    const double determinant = determinant_left - determinant_right;

    double determinant_sum = 0.0;
    if (determinant_left > 0.0) {
        if (determinant_right <= 0.0) {
            return determinant;
        }
        determinant_sum = determinant_left + determinant_right;
    } else if (determinant_left < 0.0) {
        if (determinant_right >= 0.0) {
            return determinant;
        }
        determinant_sum = -determinant_left - determinant_right;
    } else {
        return determinant;
    }

    const double error_bound = ORIENT_ERROR_BOUND_A * determinant_sum;
    if (std::abs(determinant) >= error_bound) {
        return determinant;
    }
    return expansion_orientation_value(a, b, point);
}

}  // namespace

Orientation orient2d(Point2 a, Point2 b, Point2 point) {
    return orientation_from_value(robust_orientation_value(a, b, point));
}

}  // namespace minescript_miner
