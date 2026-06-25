#pragma once

#include <minmind/config.hpp>

#include <cstddef>
#include <vector>

namespace minmind {

using TokenId = int;
using TokenIds = std::vector<TokenId>;

class MiniMindModel {
public:
    explicit MiniMindModel(MiniMindConfig config);

    const MiniMindConfig& Config() const noexcept;

    // Phase 1 placeholder: returns a deterministic logits-like vector so the
    // app/test skeleton can validate target wiring before tensor math lands.
    std::vector<float> ForwardLastToken(const TokenIds& input_ids) const;

private:
    MiniMindConfig config_;
};

struct CausalLmWeights {
    std::vector<float> token_embedding;
    std::vector<float> attn_norm_weight;
    std::vector<float> ffn_norm_weight;
    std::vector<float> final_norm_weight;
    std::vector<float> q_weight;
    std::vector<float> k_weight;
    std::vector<float> v_weight;
    std::vector<float> o_weight;
    std::vector<float> gate_weight;
    std::vector<float> up_weight;
    std::vector<float> down_weight;
    std::vector<float> lm_head_weight;
};

struct TrainStepResult {
    float loss_before = 0.0f;
    float loss_after = 0.0f;
};

std::vector<float> CausalLmForwardNoCache(
    const TokenIds& input_ids,
    const CausalLmWeights& weights,
    std::size_t vocab_size,
    std::size_t hidden_size,
    std::size_t num_heads,
    std::size_t intermediate_size,
    float rms_eps,
    float rope_theta);

float CausalLmForwardLossNoCache(
    const TokenIds& input_ids,
    const TokenIds& labels,
    const CausalLmWeights& weights,
    std::size_t vocab_size,
    std::size_t hidden_size,
    std::size_t num_heads,
    std::size_t intermediate_size,
    float rms_eps,
    float rope_theta,
    int ignore_index);

TokenIds CausalLmGenerateGreedyNoCache(
    const TokenIds& input_ids,
    const CausalLmWeights& weights,
    std::size_t vocab_size,
    std::size_t hidden_size,
    std::size_t num_heads,
    std::size_t intermediate_size,
    float rms_eps,
    float rope_theta,
    std::size_t max_new_tokens,
    int eos_token_id);

TrainStepResult CausalLmTrainLmHeadOneStepNoCache(
    const TokenIds& input_ids,
    const TokenIds& labels,
    CausalLmWeights& weights,
    std::size_t vocab_size,
    std::size_t hidden_size,
    std::size_t num_heads,
    std::size_t intermediate_size,
    float rms_eps,
    float rope_theta,
    int ignore_index,
    float learning_rate);

} // namespace minmind
