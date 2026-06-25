#include "golden_loader.hpp"
#include "test_assert.hpp"

#include <minmind/math.hpp>

#include <cmath>

namespace minmind::tests {
namespace {

void RequireClose(float actual, float expected, float tolerance)
{
    MINMIND_REQUIRE(std::isfinite(actual));
    MINMIND_REQUIRE(std::abs(actual - expected) <= tolerance);
}

} // namespace

void RopeMatchesGoldenFixture()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_rope.json");
    MINMIND_REQUIRE(fixture.name == "phase0_rope");
    MINMIND_REQUIRE(fixture.rows == 2);
    MINMIND_REQUIRE(fixture.cols == 4);

    const RopeCache rope = PrecomputeRope(
        fixture.cols,
        fixture.rows,
        static_cast<float>(fixture.rope_theta));

    MINMIND_REQUIRE(rope.cos.size() == fixture.rope_cos.size());
    MINMIND_REQUIRE(rope.sin.size() == fixture.rope_sin.size());
    for (std::size_t i = 0; i < rope.cos.size(); ++i) {
        RequireClose(rope.cos[i], fixture.rope_cos[i], 1e-5f);
        RequireClose(rope.sin[i], fixture.rope_sin[i], 1e-5f);
    }

    const std::vector<float> actual = ApplyRotary(
        fixture.rope_input,
        rope,
        fixture.rows,
        fixture.cols);

    MINMIND_REQUIRE(actual.size() == fixture.rope_output.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        RequireClose(actual[i], fixture.rope_output[i], 1e-5f);
    }
}

} // namespace minmind::tests
