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

void RequireVectorClose(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance)
{
    MINMIND_REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        RequireClose(actual[i], expected[i], tolerance);
    }
}

} // namespace

void GatedFeedForwardMatchesGoldenFixture()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_ffn.json");
    MINMIND_REQUIRE(fixture.name == "phase0_ffn");
    MINMIND_REQUIRE(fixture.rows == 2);
    MINMIND_REQUIRE(fixture.hidden_size == 3);
    MINMIND_REQUIRE(fixture.intermediate_size == 4);

    const std::vector<float> gate = LinearNoBias(
        fixture.ffn_input,
        fixture.gate_weight,
        fixture.rows,
        fixture.hidden_size,
        fixture.intermediate_size);
    RequireVectorClose(gate, fixture.gate_output, 1e-5f);

    const std::vector<float> up = LinearNoBias(
        fixture.ffn_input,
        fixture.up_weight,
        fixture.rows,
        fixture.hidden_size,
        fixture.intermediate_size);
    RequireVectorClose(up, fixture.up_output, 1e-5f);

    const std::vector<float> ffn = GatedFeedForward(
        fixture.ffn_input,
        fixture.gate_weight,
        fixture.up_weight,
        fixture.down_weight,
        fixture.rows,
        fixture.hidden_size,
        fixture.intermediate_size);
    RequireVectorClose(ffn, fixture.ffn_output, 1e-5f);
}

} // namespace minmind::tests
