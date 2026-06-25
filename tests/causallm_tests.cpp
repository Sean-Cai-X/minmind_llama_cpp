#include "golden_loader.hpp"
#include "test_assert.hpp"

#include <minmind/checkpoint.hpp>
#include <minmind/model.hpp>
#include <minmind/trainer.hpp>

#include <cmath>
#include <cstdio>
#include <string>

namespace minmind::tests {
namespace {

void RequireClose(float actual, float expected, float tolerance)
{
    MINMIND_REQUIRE(std::isfinite(actual));
    MINMIND_REQUIRE(std::abs(actual - expected) <= tolerance);
}

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

void CausalLmMatchesGoldenFixture()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_causallm.json");
    MINMIND_REQUIRE(fixture.name == "phase0_causallm");
    MINMIND_REQUIRE(fixture.vocab_size == 4);
    MINMIND_REQUIRE(fixture.seq_len == 2);
    MINMIND_REQUIRE(fixture.hidden_size == 4);
    MINMIND_REQUIRE(fixture.num_heads == 1);

    const CausalLmWeights weights = MakeWeights(fixture);

    const std::vector<float> logits = CausalLmForwardNoCache(
        fixture.input_ids,
        weights,
        fixture.vocab_size,
        fixture.hidden_size,
        fixture.num_heads,
        fixture.intermediate_size,
        static_cast<float>(fixture.eps),
        static_cast<float>(fixture.rope_theta));

    MINMIND_REQUIRE(logits.size() == fixture.logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        RequireClose(logits[i], fixture.logits[i], 1e-5f);
    }
}

void CausalLmForwardLossMatchesGoldenFixture()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_causallm.json");
    MINMIND_REQUIRE(fixture.name == "phase0_causallm");
    MINMIND_REQUIRE(fixture.labels == TokenIds({-100, 3}));

    const CausalLmWeights weights = MakeWeights(fixture);
    const float loss = CausalLmForwardLossNoCache(
        fixture.input_ids,
        fixture.labels,
        weights,
        fixture.vocab_size,
        fixture.hidden_size,
        fixture.num_heads,
        fixture.intermediate_size,
        static_cast<float>(fixture.eps),
        static_cast<float>(fixture.rope_theta),
        -100);

    MINMIND_REQUIRE(std::isfinite(loss));
    RequireClose(loss, static_cast<float>(fixture.loss), 1e-6f);
}

void CausalLmGreedyGenerationMatchesGoldenFixture()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_causallm.json");
    MINMIND_REQUIRE(fixture.name == "phase0_causallm");
    MINMIND_REQUIRE(fixture.greedy_ids == TokenIds({0, 1, 3}));

    const CausalLmWeights weights = MakeWeights(fixture);

    const TokenIds generated = CausalLmGenerateGreedyNoCache(
        fixture.input_ids,
        weights,
        fixture.vocab_size,
        fixture.hidden_size,
        fixture.num_heads,
        fixture.intermediate_size,
        static_cast<float>(fixture.eps),
        static_cast<float>(fixture.rope_theta),
        2,
        3);

    MINMIND_REQUIRE(generated == fixture.greedy_ids);
}

void CausalLmHeadOnlyTrainStepReducesLoss()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_causallm.json");
    MINMIND_REQUIRE(fixture.name == "phase0_causallm");

    CausalLmWeights weights = MakeWeights(fixture);
    const std::vector<float> original_head = weights.lm_head_weight;

    const TrainStepResult result = CausalLmTrainLmHeadOneStepNoCache(
        fixture.input_ids,
        fixture.labels,
        weights,
        fixture.vocab_size,
        fixture.hidden_size,
        fixture.num_heads,
        fixture.intermediate_size,
        static_cast<float>(fixture.eps),
        static_cast<float>(fixture.rope_theta),
        -100,
        0.25f);

    MINMIND_REQUIRE(std::isfinite(result.loss_before));
    MINMIND_REQUIRE(std::isfinite(result.loss_after));
    RequireClose(result.loss_before, static_cast<float>(fixture.loss), 1e-6f);
    MINMIND_REQUIRE(result.loss_after < result.loss_before);
    MINMIND_REQUIRE(weights.lm_head_weight != original_head);
}

void CausalLmHeadOnlyTrainLoopReducesLossHistory()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_causallm.json");
    MINMIND_REQUIRE(fixture.name == "phase0_causallm");

    CausalLmWeights weights = MakeWeights(fixture);
    const TrainLoopResult result = TrainLmHeadNoCache(
        fixture.input_ids,
        fixture.labels,
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

void CausalLmHeadCheckpointReloadReproducesLoss()
{
    const GoldenFixture fixture = LoadGoldenFixture("testdata/golden/phase0_causallm.json");
    MINMIND_REQUIRE(fixture.name == "phase0_causallm");

    CausalLmWeights trained = MakeWeights(fixture);
    const TrainStepResult train_result = CausalLmTrainLmHeadOneStepNoCache(
        fixture.input_ids,
        fixture.labels,
        trained,
        fixture.vocab_size,
        fixture.hidden_size,
        fixture.num_heads,
        fixture.intermediate_size,
        static_cast<float>(fixture.eps),
        static_cast<float>(fixture.rope_theta),
        -100,
        0.25f);

    const std::string path = "minmind_lm_head_checkpoint.tmp";
    SaveLmHeadCheckpoint(path, trained, fixture.vocab_size, fixture.hidden_size);

    CausalLmWeights reloaded = MakeWeights(fixture);
    LoadLmHeadCheckpoint(path, reloaded, fixture.vocab_size, fixture.hidden_size);
    std::remove(path.c_str());

    MINMIND_REQUIRE(reloaded.lm_head_weight.size() == trained.lm_head_weight.size());
    for (std::size_t i = 0; i < trained.lm_head_weight.size(); ++i) {
        RequireClose(reloaded.lm_head_weight[i], trained.lm_head_weight[i], 1e-7f);
    }

    const std::vector<float> trained_logits = CausalLmForwardNoCache(
        fixture.input_ids,
        trained,
        fixture.vocab_size,
        fixture.hidden_size,
        fixture.num_heads,
        fixture.intermediate_size,
        static_cast<float>(fixture.eps),
        static_cast<float>(fixture.rope_theta));
    const std::vector<float> reloaded_logits = CausalLmForwardNoCache(
        fixture.input_ids,
        reloaded,
        fixture.vocab_size,
        fixture.hidden_size,
        fixture.num_heads,
        fixture.intermediate_size,
        static_cast<float>(fixture.eps),
        static_cast<float>(fixture.rope_theta));
    MINMIND_REQUIRE(reloaded_logits.size() == trained_logits.size());
    for (std::size_t i = 0; i < trained_logits.size(); ++i) {
        RequireClose(reloaded_logits[i], trained_logits[i], 1e-6f);
    }

    const float reloaded_loss = CausalLmForwardLossNoCache(
        fixture.input_ids,
        fixture.labels,
        reloaded,
        fixture.vocab_size,
        fixture.hidden_size,
        fixture.num_heads,
        fixture.intermediate_size,
        static_cast<float>(fixture.eps),
        static_cast<float>(fixture.rope_theta),
        -100);

    RequireClose(reloaded_loss, train_result.loss_after, 1e-6f);
}

} // namespace minmind::tests
