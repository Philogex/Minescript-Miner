#include "minescript_miner/exact_geometry.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace minescript_miner {

namespace {

ExactInt exact_abs(ExactInt value) {
    return value < 0 ? -value : value;
}

ExactInt exact_gcd(ExactInt lhs, ExactInt rhs) {
    lhs = exact_abs(std::move(lhs));
    rhs = exact_abs(std::move(rhs));
    while (rhs != 0) {
        ExactInt remainder = lhs % rhs;
        lhs = std::move(rhs);
        rhs = std::move(remainder);
    }
    return lhs;
}

ExactInt common_divisor(
    const ExactInt &first,
    const ExactInt &second,
    const ExactInt &third
) {
    return exact_gcd(exact_gcd(first, second), third);
}

ExactSign sign_of(const ExactInt &value) {
    if (value > 0) {
        return ExactSign::Positive;
    }
    if (value < 0) {
        return ExactSign::Negative;
    }
    return ExactSign::Zero;
}

ExactInt round_ratio_to_even(
    const ExactInt &numerator,
    const ExactInt &denominator
) {
    ExactInt quotient = numerator / denominator;
    const ExactInt remainder = numerator % denominator;
    const ExactInt twice_remainder = remainder << 1U;
    if (
        twice_remainder > denominator ||
        (twice_remainder == denominator && (quotient & 1) != 0)
    ) {
        ++quotient;
    }
    return quotient;
}

std::int64_t binary_exponent(
    const ExactInt &numerator,
    const ExactInt &denominator
) {
    std::int64_t exponent =
        static_cast<std::int64_t>(boost::multiprecision::msb(numerator)) -
        static_cast<std::int64_t>(boost::multiprecision::msb(denominator));

    const bool below_power =
        exponent >= 0
            ? numerator < (denominator << exponent)
            : (numerator << -exponent) < denominator;
    if (below_power) {
        --exponent;
    }
    return exponent;
}

double double_from_bits(std::uint64_t bits) {
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace

ExactRational exact_rational_from_double(double value) {
    static_assert(
        std::numeric_limits<double>::is_iec559 &&
            sizeof(double) == sizeof(std::uint64_t),
        "exact double conversion requires binary64 IEEE 754"
    );
    if (!std::isfinite(value)) {
        throw std::domain_error("exact geometry requires a finite double");
    }

    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const bool negative = (bits >> 63U) != 0;
    const std::uint64_t exponent_bits = (bits >> 52U) & 0x7ffU;
    const std::uint64_t fraction_bits =
        bits & ((std::uint64_t{1} << 52U) - 1U);

    if (exponent_bits == 0 && fraction_bits == 0) {
        return ExactRational{0};
    }

    ExactInt significand = fraction_bits;
    int exponent = -1074;
    if (exponent_bits != 0) {
        significand += ExactInt{1} << 52U;
        exponent = static_cast<int>(exponent_bits) - 1023 - 52;
    }
    if (negative) {
        significand = -significand;
    }

    if (exponent >= 0) {
        return ExactRational{significand << exponent};
    }
    return ExactRational{
        std::move(significand),
        ExactInt{1} << (-exponent),
    };
}

double approximate_double(const ExactRational &value) {
    if (value.numerator() == 0) {
        return 0.0;
    }

    const bool negative = value.numerator() < 0;
    const ExactInt numerator = exact_abs(value.numerator());
    const ExactInt &denominator = value.denominator();
    const std::int64_t exponent =
        binary_exponent(numerator, denominator);

    constexpr std::int64_t min_subnormal_exponent = -1074;
    constexpr std::int64_t min_normal_exponent = -1022;
    constexpr std::int64_t max_normal_exponent = 1023;
    constexpr std::uint64_t sign_bit = std::uint64_t{1} << 63U;
    const std::uint64_t sign = negative ? sign_bit : 0;

    if (exponent < min_subnormal_exponent - 1) {
        return double_from_bits(sign);
    }
    if (exponent > max_normal_exponent) {
        return double_from_bits(
            sign | (std::uint64_t{0x7ff} << 52U)
        );
    }

    // Subnormal values share a fixed 2^-1074 unit.
    if (exponent < min_normal_exponent) {
        const ExactInt significand =
            round_ratio_to_even(
                numerator << -min_subnormal_exponent,
                denominator
            );
        if (significand == 0) {
            return double_from_bits(sign);
        }

        const std::uint64_t fraction =
            significand.convert_to<std::uint64_t>();
        return double_from_bits(sign | fraction);
    }

    // Normal values retain 53 significant bits, rounded ties-to-even.
    const std::int64_t shift = 52 - exponent;
    ExactInt significand = shift >= 0
        ? round_ratio_to_even(numerator << shift, denominator)
        : round_ratio_to_even(numerator, denominator << -shift);

    std::int64_t rounded_exponent = exponent;
    if (significand == (ExactInt{1} << 53U)) {
        significand >>= 1U;
        ++rounded_exponent;
    }
    if (rounded_exponent > max_normal_exponent) {
        return double_from_bits(
            sign | (std::uint64_t{0x7ff} << 52U)
        );
    }

    const std::uint64_t exponent_bits =
        static_cast<std::uint64_t>(rounded_exponent + 1023);
    const std::uint64_t fraction =
        (significand - (ExactInt{1} << 52U))
            .convert_to<std::uint64_t>();
    return double_from_bits(
        sign | (exponent_bits << 52U) | fraction
    );
}

ExactPoint2H normalize_exact_point(ExactPoint2H point) {
    if (!is_valid(point)) {
        return point;
    }

    const ExactInt divisor = common_divisor(point.x, point.y, point.w);
    if (divisor > 1) {
        point.x /= divisor;
        point.y /= divisor;
        point.w /= divisor;
    }

    const bool negate =
        point.w < 0 ||
        (point.w == 0 &&
         (point.x < 0 || (point.x == 0 && point.y < 0)));
    if (negate) {
        point.x = -point.x;
        point.y = -point.y;
        point.w = -point.w;
    }
    return point;
}

ExactLine2 normalize_exact_line(ExactLine2 line) {
    if (!is_valid(line)) {
        return line;
    }

    const ExactInt divisor = common_divisor(line.a, line.b, line.c);
    if (divisor > 1) {
        line.a /= divisor;
        line.b /= divisor;
        line.c /= divisor;
    }
    return line;
}

ExactPoint2H exact_point(
    const ExactRational &x,
    const ExactRational &y
) {
    return normalize_exact_point({
        x.numerator() * y.denominator(),
        y.numerator() * x.denominator(),
        x.denominator() * y.denominator(),
    });
}

ExactPoint2H exact_point(double x, double y) {
    return exact_point(
        exact_rational_from_double(x),
        exact_rational_from_double(y)
    );
}

Point2 approximate_point(const ExactPoint2H &point) {
    if (!is_finite(point)) {
        throw std::domain_error("cannot approximate a point at infinity");
    }
    return {
        approximate_double(ExactRational{point.x, point.w}),
        approximate_double(ExactRational{point.y, point.w}),
    };
}

ExactLine2 exact_line_through(
    const ExactPoint2H &from,
    const ExactPoint2H &to
) {
    return normalize_exact_line({
        from.y * to.w - from.w * to.y,
        from.w * to.x - from.x * to.w,
        from.x * to.y - from.y * to.x,
    });
}

ExactPoint2H exact_line_intersection(
    const ExactLine2 &lhs,
    const ExactLine2 &rhs
) {
    return normalize_exact_point({
        lhs.b * rhs.c - lhs.c * rhs.b,
        lhs.c * rhs.a - lhs.a * rhs.c,
        lhs.a * rhs.b - lhs.b * rhs.a,
    });
}

ExactLine2 opposite_half_plane(ExactLine2 line) {
    line.a = -line.a;
    line.b = -line.b;
    line.c = -line.c;
    return line;
}

ExactSign classify_exact(
    const ExactLine2 &line,
    const ExactPoint2H &point
) {
    return sign_of(
        line.a * point.x +
        line.b * point.y +
        line.c * point.w
    );
}

bool is_finite(const ExactPoint2H &point) {
    return point.w != 0;
}

bool is_valid(const ExactPoint2H &point) {
    return point.x != 0 || point.y != 0 || point.w != 0;
}

bool is_valid(const ExactLine2 &line) {
    return line.a != 0 || line.b != 0 || line.c != 0;
}

}  // namespace minescript_miner
