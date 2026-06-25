#pragma once

#include <cstddef>
#include <vector>

namespace minmind {

std::vector<float> RmsNorm(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::size_t rows,
    std::size_t cols,
    float eps);

struct RopeCache {
    std::size_t seq_len = 0;
    std::size_t dim = 0;
    std::vector<float> cos;
    std::vector<float> sin;
};

RopeCache PrecomputeRope(std::size_t dim, std::size_t seq_len, float theta);

std::vector<float> ApplyRotary(
    const std::vector<float>& input,
    const RopeCache& rope,
    std::size_t rows,
    std::size_t dim);

std::vector<float> LinearNoBias(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::size_t rows,
    std::size_t in_cols,
    std::size_t out_cols);

std::vector<float> Silu(const std::vector<float>& input);

std::vector<float> GatedFeedForward(
    const std::vector<float>& input,
    const std::vector<float>& gate_weight,
    const std::vector<float>& up_weight,
    const std::vector<float>& down_weight,
    std::size_t rows,
    std::size_t hidden_size,
    std::size_t intermediate_size);

std::vector<float> CausalSelfAttentionNoCache(
    const std::vector<float>& input,
    const std::vector<float>& q_weight,
    const std::vector<float>& k_weight,
    const std::vector<float>& v_weight,
    const std::vector<float>& o_weight,
    std::size_t seq_len,
    std::size_t hidden_size,
    std::size_t num_heads,
    float rope_theta);

std::vector<float> TransformerBlockNoCache(
    const std::vector<float>& input,
    const std::vector<float>& attn_norm_weight,
    const std::vector<float>& ffn_norm_weight,
    const std::vector<float>& q_weight,
    const std::vector<float>& k_weight,
    const std::vector<float>& v_weight,
    const std::vector<float>& o_weight,
    const std::vector<float>& gate_weight,
    const std::vector<float>& up_weight,
    const std::vector<float>& down_weight,
    std::size_t seq_len,
    std::size_t hidden_size,
    std::size_t num_heads,
    std::size_t intermediate_size,
    float rms_eps,
    float rope_theta);

float MeanCrossEntropyLoss(
    const std::vector<float>& logits,
    const std::vector<int>& labels,
    std::size_t rows,
    std::size_t vocab_size,
    int ignore_index);

} // namespace minmind
