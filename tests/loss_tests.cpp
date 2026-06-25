#include "golden_loader.hpp"
#include "test_assert.hpp"

#include <minmind/math.hpp>

#include <cmath>

namespace minmind::tests {

void MeanCrossEntropyLossMatchesGoldenFixture()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_loss.json");
    MINMIND_REQUIRE(fixture.name == "phase0_loss");
    MINMIND_REQUIRE(fixture.rows == 2);
    MINMIND_REQUIRE(fixture.vocab_size == 4);
    MINMIND_REQUIRE(fixture.labels == TokenIds({-100, 3}));

    const float loss = MeanCrossEntropyLoss(
        fixture.logits,
        fixture.labels,
        fixture.rows,
        fixture.vocab_size,
        -100);

    MINMIND_REQUIRE(std::isfinite(loss));
    MINMIND_REQUIRE(std::abs(loss - static_cast<float>(fixture.loss)) <= 1e-6f);
}

} // namespace minmind::tests
