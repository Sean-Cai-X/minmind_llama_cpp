#include "golden_loader.hpp"
#include "test_assert.hpp"

namespace minmind::tests {

void GoldenFixtureLoadsTokenIds()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_token_ids.json");
    MINMIND_REQUIRE(fixture.name == "phase0_token_ids");
    MINMIND_REQUIRE(fixture.prompt == "1 10 11 12");
    MINMIND_REQUIRE(fixture.input_ids == TokenIds({1, 10, 11, 12}));
    MINMIND_REQUIRE(fixture.labels == TokenIds({-100, -100, 11, 12}));
    MINMIND_REQUIRE(fixture.attention_mask == TokenIds({1, 1, 1, 1}));
    MINMIND_REQUIRE(fixture.greedy_ids == TokenIds({1, 10, 11, 12, 13}));
}

} // namespace minmind::tests
