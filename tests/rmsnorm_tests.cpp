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

void RmsNormMatchesGoldenFixture()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_rmsnorm.json");
    MINMIND_REQUIRE(fixture.name == "phase0_rmsnorm");
    MINMIND_REQUIRE(fixture.rows == 2);
    MINMIND_REQUIRE(fixture.cols == 4);

    const std::vector<float> actual = RmsNorm(
        fixture.rmsnorm_input,
        fixture.rmsnorm_weight,
        fixture.rows,
        fixture.cols,
        static_cast<float>(fixture.eps));

    MINMIND_REQUIRE(actual.size() == fixture.rmsnorm_output.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        RequireClose(actual[i], fixture.rmsnorm_output[i], 1e-5f);
    }
}

} // namespace minmind::tests
