#pragma once

#include <minmind/model.hpp>

#include <string>
#include <vector>

namespace minmind::tests {

struct GoldenFixture {
    std::string name;
    std::string prompt;
    TokenIds input_ids;
    TokenIds labels;
    TokenIds attention_mask;
    TokenIds greedy_ids;
    double loss = 0.0;
    double eps = 0.0;
    double rope_theta = 0.0;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t seq_len = 0;
    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t intermediate_size = 0;
    std::size_t num_heads = 0;
    std::vector<float> logits_last;
    std::vector<float> rmsnorm_input;
    std::vector<float> rmsnorm_weight;
    std::vector<float> rmsnorm_output;
    std::vector<float> rope_input;
    std::vector<float> rope_cos;
    std::vector<float> rope_sin;
    std::vector<float> rope_output;
    std::vector<float> ffn_input;
    std::vector<float> gate_weight;
    std::vector<float> up_weight;
    std::vector<float> down_weight;
    std::vector<float> gate_output;
    std::vector<float> up_output;
    std::vector<float> ffn_output;
    std::vector<float> attention_input;
    std::vector<float> q_weight;
    std::vector<float> k_weight;
    std::vector<float> v_weight;
    std::vector<float> o_weight;
    std::vector<float> attention_output;
    std::vector<float> block_input;
    std::vector<float> attn_norm_weight;
    std::vector<float> ffn_norm_weight;
    std::vector<float> final_norm_weight;
    std::vector<float> block_attention_output;
    std::vector<float> block_output;
    std::vector<float> token_embedding;
    std::vector<float> lm_head_weight;
    std::vector<float> logits;
};

GoldenFixture LoadGoldenFixture(const std::string& relative_path);

} // namespace minmind::tests
