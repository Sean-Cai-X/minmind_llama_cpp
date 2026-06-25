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

void CausalSelfAttentionMatchesGoldenFixture()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_attention.json");
    MINMIND_REQUIRE(fixture.name == "phase0_attention");
    MINMIND_REQUIRE(fixture.seq_len == 2);
    MINMIND_REQUIRE(fixture.hidden_size == 4);
    MINMIND_REQUIRE(fixture.num_heads == 1);

    const std::vector<float> actual = CausalSelfAttentionNoCache(
        fixture.attention_input,
        fixture.q_weight,
        fixture.k_weight,
        fixture.v_weight,
        fixture.o_weight,
        fixture.seq_len,
        fixture.hidden_size,
        fixture.num_heads,
        static_cast<float>(fixture.rope_theta));

    MINMIND_REQUIRE(actual.size() == fixture.attention_output.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        RequireClose(actual[i], fixture.attention_output[i], 1e-5f);
    }
}

} // namespace minmind::tests
