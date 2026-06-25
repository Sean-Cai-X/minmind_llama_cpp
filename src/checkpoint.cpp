#include <minmind/checkpoint.hpp>

#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>

namespace minmind {
namespace {

constexpr const char* kLmHeadCheckpointMagic = "MINMIND_LM_HEAD_V1";

void ValidateShape(
    const CausalLmWeights& weights,
    std::size_t vocab_size,
    std::size_t hidden_size)
{
    if (vocab_size == 0 || hidden_size == 0) {
        throw std::invalid_argument("checkpoint dimensions must be greater than zero");
    }
    if (weights.lm_head_weight.size() != vocab_size * hidden_size) {
        throw std::invalid_argument("lm_head_weight size does not match checkpoint shape");
    }
}

} // namespace

void SaveLmHeadCheckpoint(
    const std::string& path,
    const CausalLmWeights& weights,
    std::size_t vocab_size,
    std::size_t hidden_size)
{
    ValidateShape(weights, vocab_size, hidden_size);

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open checkpoint for writing: " + path);
    }

    output << kLmHeadCheckpointMagic << '\n';
    output << vocab_size << ' ' << hidden_size << '\n';
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t i = 0; i < weights.lm_head_weight.size(); ++i) {
        if (i != 0) {
            output << ' ';
        }
        output << weights.lm_head_weight[i];
    }
    output << '\n';

    if (!output) {
        throw std::runtime_error("failed to write checkpoint: " + path);
    }
}

void LoadLmHeadCheckpoint(
    const std::string& path,
    CausalLmWeights& weights,
    std::size_t vocab_size,
    std::size_t hidden_size)
{
    if (vocab_size == 0 || hidden_size == 0) {
        throw std::invalid_argument("checkpoint dimensions must be greater than zero");
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open checkpoint for reading: " + path);
    }

    std::string magic;
    input >> magic;
    if (magic != kLmHeadCheckpointMagic) {
        throw std::runtime_error("invalid LM head checkpoint magic");
    }

    std::size_t stored_vocab_size = 0;
    std::size_t stored_hidden_size = 0;
    input >> stored_vocab_size >> stored_hidden_size;
    if (stored_vocab_size != vocab_size || stored_hidden_size != hidden_size) {
        throw std::runtime_error("LM head checkpoint shape does not match requested shape");
    }

    std::vector<float> loaded(vocab_size * hidden_size, 0.0f);
    for (float& value : loaded) {
        input >> value;
    }
    if (!input) {
        throw std::runtime_error("LM head checkpoint ended before all weights were read");
    }

    weights.lm_head_weight = std::move(loaded);
}

} // namespace minmind
