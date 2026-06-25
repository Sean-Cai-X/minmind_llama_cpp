#include <minmind/config.hpp>

namespace minmind {

bool IsValid(const MiniMindConfig& config, std::string* error)
{
    auto fail = [error](const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };

    if (config.vocab_size == 0) {
        return fail("vocab_size must be greater than zero");
    }
    if (config.hidden_size == 0) {
        return fail("hidden_size must be greater than zero");
    }
    if (config.num_hidden_layers == 0) {
        return fail("num_hidden_layers must be greater than zero");
    }
    if (config.num_attention_heads == 0) {
        return fail("num_attention_heads must be greater than zero");
    }
    if (config.hidden_size % config.num_attention_heads != 0) {
        return fail("hidden_size must be divisible by num_attention_heads");
    }
    if (config.num_key_value_heads == 0) {
        return fail("num_key_value_heads must be greater than zero");
    }
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        return fail("num_attention_heads must be divisible by num_key_value_heads");
    }
    if (config.max_seq_len == 0) {
        return fail("max_seq_len must be greater than zero");
    }
    if (config.pad_token_id < 0 || config.bos_token_id < 0 || config.eos_token_id < 0) {
        return fail("special token ids must be non-negative");
    }

    if (error != nullptr) {
        error->clear();
    }
    return true;
}

} // namespace minmind
