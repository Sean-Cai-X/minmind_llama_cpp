#include <minmind/model.hpp>

#include <minmind/math.hpp>
#include <minmind/sampling.hpp>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <stdexcept>

namespace minmind {
namespace {

std::vector<float> CausalLmHiddenNoCache(
    const TokenIds& input_ids,
    const CausalLmWeights& weights,
    std::size_t vocab_size,
    std::size_t hidden_size,
    std::size_t num_heads,
    std::size_t intermediate_size,
    float rms_eps,
    float rope_theta)
{
    if (input_ids.empty()) {
        throw std::invalid_argument("CausalLm input_ids must not be empty");
    }
    if (vocab_size == 0 || hidden_size == 0) {
        throw std::invalid_argument("CausalLm vocab_size and hidden_size must be greater than zero");
    }
    if (weights.token_embedding.size() != vocab_size * hidden_size) {
        throw std::invalid_argument("token_embedding size does not match vocab_size * hidden_size");
    }

    const std::size_t seq_len = input_ids.size();
    std::vector<float> hidden(seq_len * hidden_size, 0.0f);
    for (std::size_t t = 0; t < seq_len; ++t) {
        const int token = input_ids[t];
        if (token < 0 || static_cast<std::size_t>(token) >= vocab_size) {
            throw std::invalid_argument("input token id is outside vocabulary");
        }
        const std::size_t token_offset = static_cast<std::size_t>(token) * hidden_size;
        for (std::size_t d = 0; d < hidden_size; ++d) {
            hidden[t * hidden_size + d] = weights.token_embedding[token_offset + d];
        }
    }

    hidden = TransformerBlockNoCache(
        hidden,
        weights.attn_norm_weight,
        weights.ffn_norm_weight,
        weights.q_weight,
        weights.k_weight,
        weights.v_weight,
        weights.o_weight,
        weights.gate_weight,
        weights.up_weight,
        weights.down_weight,
        seq_len,
        hidden_size,
        num_heads,
        intermediate_size,
        rms_eps,
        rope_theta);

    return RmsNorm(hidden, weights.final_norm_weight, seq_len, hidden_size, rms_eps);
}

} // namespace

MiniMindModel::MiniMindModel(MiniMindConfig config) : config_(config)
{
    std::string error;
    if (!IsValid(config_, &error)) {
        throw std::invalid_argument(error);
    }
}

const MiniMindConfig& MiniMindModel::Config() const noexcept
{
    return config_;
}

std::vector<float> MiniMindModel::ForwardLastToken(const TokenIds& input_ids) const
{
    std::vector<float> logits(config_.vocab_size, 0.0f);
    if (input_ids.empty()) {
        logits[static_cast<std::size_t>(config_.eos_token_id)] = 1.0f;
        return logits;
    }

    const int last = input_ids.back();
    const std::size_t next =
        static_cast<std::size_t>((last + 1) % static_cast<int>(config_.vocab_size));
    logits[next] = 1.0f;
    logits[static_cast<std::size_t>(config_.eos_token_id)] = 0.5f;
    return logits;
}

std::vector<float> CausalLmForwardNoCache(
    const TokenIds& input_ids,
    const CausalLmWeights& weights,
    std::size_t vocab_size,
    std::size_t hidden_size,
    std::size_t num_heads,
    std::size_t intermediate_size,
    float rms_eps,
    float rope_theta)
{
    if (input_ids.empty()) {
        throw std::invalid_argument("CausalLmForwardNoCache input_ids must not be empty");
    }
    if (vocab_size == 0 || hidden_size == 0) {
        throw std::invalid_argument("CausalLmForwardNoCache vocab_size and hidden_size must be greater than zero");
    }
    if (weights.lm_head_weight.size() != vocab_size * hidden_size) {
        throw std::invalid_argument("lm_head_weight size does not match vocab_size * hidden_size");
    }

    const std::size_t seq_len = input_ids.size();
    const std::vector<float> hidden = CausalLmHiddenNoCache(
        input_ids,
        weights,
        vocab_size,
        hidden_size,
        num_heads,
        intermediate_size,
        rms_eps,
        rope_theta);
    return LinearNoBias(hidden, weights.lm_head_weight, seq_len, hidden_size, vocab_size);
}

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
    int ignore_index)
{
    const std::vector<float> logits = CausalLmForwardNoCache(
        input_ids,
        weights,
        vocab_size,
        hidden_size,
        num_heads,
        intermediate_size,
        rms_eps,
        rope_theta);
    return MeanCrossEntropyLoss(logits, labels, input_ids.size(), vocab_size, ignore_index);
}

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
    int eos_token_id)
{
    if (input_ids.empty()) {
        throw std::invalid_argument("CausalLmGenerateGreedyNoCache input_ids must not be empty");
    }

    TokenIds generated = input_ids;
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        const std::vector<float> logits = CausalLmForwardNoCache(
            generated,
            weights,
            vocab_size,
            hidden_size,
            num_heads,
            intermediate_size,
            rms_eps,
            rope_theta);

        const std::vector<float> last_logits(
            logits.end() - static_cast<std::ptrdiff_t>(vocab_size),
            logits.end());
        const int next_token = GreedySample(last_logits);
        generated.push_back(next_token);
        if (next_token == eos_token_id) {
            break;
        }
    }
    return generated;
}

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
    float learning_rate)
{
    if (learning_rate <= 0.0f) {
        throw std::invalid_argument("learning_rate must be greater than zero");
    }
    if (weights.lm_head_weight.size() != vocab_size * hidden_size) {
        throw std::invalid_argument("lm_head_weight size does not match vocab_size * hidden_size");
    }

    const std::size_t seq_len = input_ids.size();
    const std::vector<float> hidden = CausalLmHiddenNoCache(
        input_ids,
        weights,
        vocab_size,
        hidden_size,
        num_heads,
        intermediate_size,
        rms_eps,
        rope_theta);
    const std::vector<float> logits = LinearNoBias(
        hidden, weights.lm_head_weight, seq_len, hidden_size, vocab_size);

    TrainStepResult result;
    result.loss_before = MeanCrossEntropyLoss(logits, labels, seq_len, vocab_size, ignore_index);

    std::size_t contributing_labels = 0;
    for (const int label : labels) {
        if (label != ignore_index) {
            ++contributing_labels;
        }
    }
    if (contributing_labels == 0) {
        throw std::invalid_argument("training step requires at least one non-ignored label");
    }

    std::vector<float> grad(weights.lm_head_weight.size(), 0.0f);
    const double normalizer = 1.0 / static_cast<double>(contributing_labels);
    for (std::size_t row = 0; row < seq_len; ++row) {
        const int label = labels[row];
        if (label == ignore_index) {
            continue;
        }
        if (label < 0 || static_cast<std::size_t>(label) >= vocab_size) {
            throw std::invalid_argument("training label is outside vocabulary");
        }

        const std::size_t logits_offset = row * vocab_size;
        double max_logit = static_cast<double>(logits[logits_offset]);
        for (std::size_t vocab = 1; vocab < vocab_size; ++vocab) {
            max_logit = std::max(max_logit, static_cast<double>(logits[logits_offset + vocab]));
        }

        double exp_sum = 0.0;
        std::vector<double> probabilities(vocab_size, 0.0);
        for (std::size_t vocab = 0; vocab < vocab_size; ++vocab) {
            probabilities[vocab] = std::exp(static_cast<double>(logits[logits_offset + vocab]) - max_logit);
            exp_sum += probabilities[vocab];
        }

        for (std::size_t vocab = 0; vocab < vocab_size; ++vocab) {
            double grad_logit = probabilities[vocab] / exp_sum;
            if (vocab == static_cast<std::size_t>(label)) {
                grad_logit -= 1.0;
            }
            grad_logit *= normalizer;

            for (std::size_t hidden_col = 0; hidden_col < hidden_size; ++hidden_col) {
                grad[vocab * hidden_size + hidden_col] +=
                    static_cast<float>(grad_logit * static_cast<double>(hidden[row * hidden_size + hidden_col]));
            }
        }
    }

    for (std::size_t i = 0; i < weights.lm_head_weight.size(); ++i) {
        weights.lm_head_weight[i] -= learning_rate * grad[i];
    }

    result.loss_after = CausalLmForwardLossNoCache(
        input_ids,
        labels,
        weights,
        vocab_size,
        hidden_size,
        num_heads,
        intermediate_size,
        rms_eps,
        rope_theta,
        ignore_index);
    return result;
}

} // namespace minmind
