#include "golden_loader.hpp"
#include "test_assert.hpp"

#include <minmind/dataset.hpp>
#include <minmind/trainer.hpp>

#include <cmath>
#include <string>

#ifndef MINMIND_SOURCE_DIR
#define MINMIND_SOURCE_DIR "."
#endif

namespace minmind::tests {
namespace {

CausalLmWeights MakeWeights(const GoldenFixture& fixture)
{
    CausalLmWeights weights;
    weights.token_embedding = fixture.token_embedding;
    weights.attn_norm_weight = fixture.attn_norm_weight;
    weights.ffn_norm_weight = fixture.ffn_norm_weight;
    weights.final_norm_weight = fixture.final_norm_weight;
    weights.q_weight = fixture.q_weight;
    weights.k_weight = fixture.k_weight;
    weights.v_weight = fixture.v_weight;
    weights.o_weight = fixture.o_weight;
    weights.gate_weight = fixture.gate_weight;
    weights.up_weight = fixture.up_weight;
    weights.down_weight = fixture.down_weight;
    weights.lm_head_weight = fixture.lm_head_weight;
    return weights;
}

} // namespace

void JsonlTrainSamplesLoadTokenIdsAndLabels()
{
    const std::string path = std::string(MINMIND_SOURCE_DIR) + "/testdata/train/tiny_lm.jsonl";
    const std::vector<TrainSample> samples = LoadJsonlTrainSamples(path);

    MINMIND_REQUIRE(samples.size() == 1);
    MINMIND_REQUIRE(samples[0].input_ids == TokenIds({0, 1}));
    MINMIND_REQUIRE(samples[0].labels == TokenIds({-100, 3}));
}

void NextTokenTrainSamplesUseFixedLengthWindows()
{
    const TokenIds ids = {0, 1, 2, 3, 0, 1, 2};
    const std::vector<TrainSample> samples = BuildNextTokenTrainSamples(ids, 3, 2);

    MINMIND_REQUIRE(samples.size() == 2);
    MINMIND_REQUIRE(samples[0].input_ids == TokenIds({0, 1, 2}));
    MINMIND_REQUIRE(samples[0].labels == TokenIds({1, 2, 3}));
    MINMIND_REQUIRE(samples[1].input_ids == TokenIds({3, 0, 1}));
    MINMIND_REQUIRE(samples[1].labels == TokenIds({0, 1, 2}));
}

void JsonlTrainSamplesDriveTinyTrainLoop()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_causallm.json");
    const std::string path = std::string(MINMIND_SOURCE_DIR) + "/testdata/train/tiny_lm.jsonl";
    const std::vector<TrainSample> samples = LoadJsonlTrainSamples(path);

    CausalLmWeights weights = MakeWeights(fixture);
    const TrainLoopResult result = TrainLmHeadSamplesNoCache(
        samples,
        weights,
        fixture.vocab_size,
        fixture.hidden_size,
        fixture.num_heads,
        fixture.intermediate_size,
        static_cast<float>(fixture.eps),
        static_cast<float>(fixture.rope_theta),
        -100,
        0.25f,
        4);

    MINMIND_REQUIRE(result.loss_history.size() == 5);
    for (std::size_t i = 0; i < result.loss_history.size(); ++i) {
        MINMIND_REQUIRE(std::isfinite(result.loss_history[i]));
        if (i > 0) {
            MINMIND_REQUIRE(result.loss_history[i] < result.loss_history[i - 1]);
        }
    }
}

} // namespace minmind::tests
