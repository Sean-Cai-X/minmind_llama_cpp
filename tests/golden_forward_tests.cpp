#include "golden_loader.hpp"
#include "test_assert.hpp"

#include <cmath>

namespace minmind::tests {

void GoldenForwardFixtureLoadsLossAndLogits()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_forward.json");
    MINMIND_REQUIRE(fixture.name == "phase0_forward");
    MINMIND_REQUIRE(std::isfinite(fixture.loss));
    MINMIND_REQUIRE(fixture.loss >= 0.0);
    MINMIND_REQUIRE(!fixture.logits_last.empty());
    MINMIND_REQUIRE(fixture.greedy_ids.size() == fixture.input_ids.size() + 1);
}

} // namespace minmind::tests
