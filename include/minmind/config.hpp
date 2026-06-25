#pragma once

#include <cstddef>
#include <string>

namespace minmind {

struct MiniMindConfig {
    std::size_t vocab_size = 6400;
    std::size_t hidden_size = 64;
    std::size_t num_hidden_layers = 2;
    std::size_t num_attention_heads = 4;
    std::size_t num_key_value_heads = 2;
    std::size_t max_seq_len = 128;
    int bos_token_id = 1;
    int eos_token_id = 2;
    int pad_token_id = 0;
    bool use_moe = false;
};

bool IsValid(const MiniMindConfig& config, std::string* error = nullptr);

} // namespace minmind
