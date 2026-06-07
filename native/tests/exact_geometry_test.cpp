#include "minescript_miner/exact_geometry.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

bool same_bits(double lhs, double rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(double)) == 0;
}

double from_bits(std::uint64_t bits) {
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace

int main() {
    using namespace minescript_miner;

    const double samples[]{
        0.0,
        -0.0,
        0.1,
        -13.2,
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::min(),
        std::numeric_limits<double>::max(),
    };
    for (const double sample : samples) {
        const double round_trip =
            approximate_double(exact_rational_from_double(sample));
        const bool matches =
            sample == 0.0 ? round_trip == 0.0 : same_bits(sample, round_trip);
        if (!matches) {
            std::cerr << "exact double round trip failed for "
                      << std::hexfloat << sample << ": got "
                      << round_trip << '\n';
        }
        assert(matches);
    }
    assert((exact_rational_from_double(0.1) != ExactRational{1, 10}));

    std::uint64_t random_bits = 0x243f6a8885a308d3ULL;
    for (int index = 0; index < 2048; ++index) {
        random_bits =
            random_bits * 6364136223846793005ULL +
            1442695040888963407ULL;
        if (((random_bits >> 52U) & 0x7ffU) == 0x7ffU) {
            continue;
        }
        const double sample = from_bits(random_bits);
        const double round_trip =
            approximate_double(exact_rational_from_double(sample));
        assert(sample == 0.0 || same_bits(sample, round_trip));
    }

    const ExactInt subnormal_denominator = ExactInt{1} << 1075U;
    assert(
        same_bits(
            approximate_double(
                ExactRational{ExactInt{1}, subnormal_denominator}
            ),
            0.0
        )
    );
    assert(
        same_bits(
            approximate_double(
                ExactRational{ExactInt{3}, subnormal_denominator}
            ),
            from_bits(2)
        )
    );
    assert(
        same_bits(
            approximate_double(
                ExactRational{
                    (ExactInt{1} << 53U) + 1,
                    ExactInt{1} << 53U,
                }
            ),
            1.0
        )
    );

    bool rejected_non_finite = false;
    try {
        (void) exact_rational_from_double(
            std::numeric_limits<double>::infinity()
        );
    } catch (const std::domain_error &) {
        rejected_non_finite = true;
    }
    assert(rejected_non_finite);

    const ExactRational one_third{1, 3};
    const ExactRational two_fifths{2, 5};
    const ExactPoint2H vertical_a = exact_point(one_third, ExactRational{0});
    const ExactPoint2H vertical_b = exact_point(one_third, ExactRational{1});
    const ExactPoint2H horizontal_a =
        exact_point(ExactRational{0}, two_fifths);
    const ExactPoint2H horizontal_b =
        exact_point(ExactRational{1}, two_fifths);

    const ExactLine2 vertical = exact_line_through(vertical_a, vertical_b);
    const ExactLine2 horizontal =
        exact_line_through(horizontal_b, horizontal_a);
    const ExactPoint2H intersection =
        exact_line_intersection(vertical, horizontal);
    assert(intersection == exact_point(one_third, two_fifths));
    assert(classify_exact(vertical, intersection) == ExactSign::Zero);
    assert(
        classify_exact(vertical, exact_point(0.0, 0.0)) ==
        ExactSign::Positive
    );
    assert(
        classify_exact(
            opposite_half_plane(vertical),
            exact_point(0.0, 0.0)
        ) == ExactSign::Negative
    );

    const ExactLine2 normalized =
        normalize_exact_line({vertical.a * 7, vertical.b * 7, vertical.c * 7});
    assert(normalized == vertical);
    assert(
        normalize_exact_point({
            intersection.x * -11,
            intersection.y * -11,
            intersection.w * -11,
        }) == intersection
    );
    assert(!is_valid(exact_line_through(vertical_a, vertical_a)));
    assert(!is_valid(exact_line_intersection(vertical, vertical)));

    const ExactLine2 parallel = exact_line_through(
        exact_point(ExactRational{2, 3}, ExactRational{0}),
        exact_point(ExactRational{2, 3}, ExactRational{1})
    );
    const ExactPoint2H point_at_infinity =
        exact_line_intersection(vertical, parallel);
    assert(is_valid(point_at_infinity));
    assert(!is_finite(point_at_infinity));

    const Point2 approximate = approximate_point(intersection);
    assert(std::abs(approximate.x - 1.0 / 3.0) < 1e-15);
    assert(std::abs(approximate.y - 2.0 / 5.0) < 1e-15);
}
