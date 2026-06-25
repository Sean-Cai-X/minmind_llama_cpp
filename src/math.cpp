#include <minmind/math.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace minmind {

std::vector<float> RmsNorm(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::size_t rows,
    std::size_t cols,
    float eps)
{
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("RmsNorm rows and cols must be greater than zero");
    }
    if (input.size() != rows * cols) {
        throw std::invalid_argument("RmsNorm input size does not match rows * cols");
    }
    if (weight.size() != cols) {
        throw std::invalid_argument("RmsNorm weight size must match cols");
    }
    if (eps < 0.0f) {
        throw std::invalid_argument("RmsNorm eps must be non-negative");
    }

    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        double mean_square = 0.0;
        const std::size_t offset = row * cols;
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = static_cast<double>(input[offset + col]);
            mean_square += value * value;
        }
        mean_square /= static_cast<double>(cols);
        const double scale = 1.0 / std::sqrt(mean_square + static_cast<double>(eps));

        for (std::size_t col = 0; col < cols; ++col) {
            output[offset + col] =
                static_cast<float>(static_cast<double>(input[offset + col]) *
                                   scale *
                                   static_cast<double>(weight[col]));
        }
    }
    return output;
}

RopeCache PrecomputeRope(std::size_t dim, std::size_t seq_len, float theta)
{
    if (dim == 0 || dim % 2 != 0) {
        throw std::invalid_argument("RoPE dim must be a positive even number");
    }
    if (seq_len == 0) {
        throw std::invalid_argument("RoPE seq_len must be greater than zero");
    }
    if (theta <= 0.0f) {
        throw std::invalid_argument("RoPE theta must be greater than zero");
    }

    RopeCache cache;
    cache.seq_len = seq_len;
    cache.dim = dim;
    cache.cos.resize(seq_len * dim);
    cache.sin.resize(seq_len * dim);

    for (std::size_t pos = 0; pos < seq_len; ++pos) {
        for (std::size_t i = 0; i < dim / 2; ++i) {
            const double exponent = static_cast<double>(2 * i) / static_cast<double>(dim);
            const double inv_freq = 1.0 / std::pow(static_cast<double>(theta), exponent);
            const double angle = static_cast<double>(pos) * inv_freq;
            const float c = static_cast<float>(std::cos(angle));
            const float s = static_cast<float>(std::sin(angle));
            cache.cos[pos * dim + i] = c;
            cache.cos[pos * dim + i + dim / 2] = c;
            cache.sin[pos * dim + i] = s;
            cache.sin[pos * dim + i + dim / 2] = s;
        }
    }
    return cache;
}

std::vector<float> ApplyRotary(
    const std::vector<float>& input,
    const RopeCache& rope,
    std::size_t rows,
    std::size_t dim)
{
    if (rows == 0 || dim == 0 || dim % 2 != 0) {
        throw std::invalid_argument("ApplyRotary rows and even dim must be greater than zero");
    }
    if (input.size() != rows * dim) {
        throw std::invalid_argument("ApplyRotary input size does not match rows * dim");
    }
    if (rope.dim != dim || rope.seq_len < rows) {
        throw std::invalid_argument("ApplyRotary rope cache shape does not match input");
    }
    if (rope.cos.size() < rows * dim || rope.sin.size() < rows * dim) {
        throw std::invalid_argument("ApplyRotary rope cache buffers are too small");
    }

    std::vector<float> output(input.size(), 0.0f);
    const std::size_t half = dim / 2;
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t offset = row * dim;
        for (std::size_t col = 0; col < dim; ++col) {
            const std::size_t rotated_col = col < half ? col + half : col - half;
            const float rotated = col < half ? -input[offset + rotated_col] : input[offset + rotated_col];
            output[offset + col] =
                input[offset + col] * rope.cos[offset + col] +
                rotated * rope.sin[offset + col];
        }
    }
    return output;
}

std::vector<float> LinearNoBias(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::size_t rows,
    std::size_t in_cols,
    std::size_t out_cols)
{
    if (rows == 0 || in_cols == 0 || out_cols == 0) {
        throw std::invalid_argument("LinearNoBias dimensions must be greater than zero");
    }
    if (input.size() != rows * in_cols) {
        throw std::invalid_argument("LinearNoBias input size does not match rows * in_cols");
    }
    if (weight.size() != out_cols * in_cols) {
        throw std::invalid_argument("LinearNoBias weight size does not match out_cols * in_cols");
    }

    std::vector<float> output(rows * out_cols, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < out_cols; ++out) {
            double sum = 0.0;
            for (std::size_t in = 0; in < in_cols; ++in) {
                sum += static_cast<double>(input[row * in_cols + in]) *
                       static_cast<double>(weight[out * in_cols + in]);
            }
            output[row * out_cols + out] = static_cast<float>(sum);
        }
    }
    return output;
}

std::vector<float> Silu(const std::vector<float>& input)
{
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const double x = static_cast<double>(input[i]);
        output[i] = static_cast<float>(x / (1.0 + std::exp(-x)));
    }
    return output;
}

std::vector<float> GatedFeedForward(
    const std::vector<float>& input,
    const std::vector<float>& gate_weight,
    const std::vector<float>& up_weight,
    const std::vector<float>& down_weight,
    std::size_t rows,
    std::size_t hidden_size,
    std::size_t intermediate_size)
{
    const std::vector<float> gate = LinearNoBias(
        input, gate_weight, rows, hidden_size, intermediate_size);
    const std::vector<float> up = LinearNoBias(
        input, up_weight, rows, hidden_size, intermediate_size);
    const std::vector<float> gate_activated = Silu(gate);

    std::vector<float> multiplied(rows * intermediate_size, 0.0f);
    for (std::size_t i = 0; i < multiplied.size(); ++i) {
        multiplied[i] = gate_activated[i] * up[i];
    }

    return LinearNoBias(multiplied, down_weight, rows, intermediate_size, hidden_size);
}

std::vector<float> CausalSelfAttentionNoCache(
    const std::vector<float>& input,
    const std::vector<float>& q_weight,
    const std::vector<float>& k_weight,
    const std::vector<float>& v_weight,
    const std::vector<float>& o_weight,
    std::size_t seq_len,
    std::size_t hidden_size,
    std::size_t num_heads,
    float rope_theta)
{
    if (seq_len == 0 || hidden_size == 0 || num_heads == 0) {
        throw std::invalid_argument("attention dimensions must be greater than zero");
    }
    if (hidden_size % num_heads != 0) {
        throw std::invalid_argument("hidden_size must be divisible by num_heads");
    }

    const std::size_t head_dim = hidden_size / num_heads;
    const std::vector<float> q = LinearNoBias(input, q_weight, seq_len, hidden_size, hidden_size);
    const std::vector<float> k = LinearNoBias(input, k_weight, seq_len, hidden_size, hidden_size);
    const std::vector<float> v = LinearNoBias(input, v_weight, seq_len, hidden_size, hidden_size);
    const RopeCache rope = PrecomputeRope(head_dim, seq_len, rope_theta);

    std::vector<float> q_rot(q.size(), 0.0f);
    std::vector<float> k_rot(k.size(), 0.0f);
    for (std::size_t head = 0; head < num_heads; ++head) {
        std::vector<float> q_head(seq_len * head_dim, 0.0f);
        std::vector<float> k_head(seq_len * head_dim, 0.0f);
        for (std::size_t t = 0; t < seq_len; ++t) {
            for (std::size_t d = 0; d < head_dim; ++d) {
                const std::size_t src = t * hidden_size + head * head_dim + d;
                const std::size_t dst = t * head_dim + d;
                q_head[dst] = q[src];
                k_head[dst] = k[src];
            }
        }

        q_head = ApplyRotary(q_head, rope, seq_len, head_dim);
        k_head = ApplyRotary(k_head, rope, seq_len, head_dim);

        for (std::size_t t = 0; t < seq_len; ++t) {
            for (std::size_t d = 0; d < head_dim; ++d) {
                const std::size_t dst = t * hidden_size + head * head_dim + d;
                const std::size_t src = t * head_dim + d;
                q_rot[dst] = q_head[src];
                k_rot[dst] = k_head[src];
            }
        }
    }

    std::vector<float> context(seq_len * hidden_size, 0.0f);
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    for (std::size_t head = 0; head < num_heads; ++head) {
        for (std::size_t t = 0; t < seq_len; ++t) {
            std::vector<double> scores(t + 1, 0.0);
            for (std::size_t s = 0; s <= t; ++s) {
                double dot = 0.0;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    const std::size_t q_idx = t * hidden_size + head * head_dim + d;
                    const std::size_t k_idx = s * hidden_size + head * head_dim + d;
                    dot += static_cast<double>(q_rot[q_idx]) * static_cast<double>(k_rot[k_idx]);
                }
                scores[s] = dot * scale;
            }

            const double max_score = *std::max_element(scores.begin(), scores.end());
            double exp_sum = 0.0;
            for (double& score : scores) {
                score = std::exp(score - max_score);
                exp_sum += score;
            }

            for (std::size_t d = 0; d < head_dim; ++d) {
                double value = 0.0;
                for (std::size_t s = 0; s <= t; ++s) {
                    const double prob = scores[s] / exp_sum;
                    const std::size_t v_idx = s * hidden_size + head * head_dim + d;
                    value += prob * static_cast<double>(v[v_idx]);
                }
                context[t * hidden_size + head * head_dim + d] = static_cast<float>(value);
            }
        }
    }

    return LinearNoBias(context, o_weight, seq_len, hidden_size, hidden_size);
}

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
    float rope_theta)
{
    if (input.size() != seq_len * hidden_size) {
        throw std::invalid_argument("TransformerBlock input size does not match seq_len * hidden_size");
    }

    const std::vector<float> attn_input = RmsNorm(
        input, attn_norm_weight, seq_len, hidden_size, rms_eps);
    const std::vector<float> attn_output = CausalSelfAttentionNoCache(
        attn_input,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        seq_len,
        hidden_size,
        num_heads,
        rope_theta);

    std::vector<float> residual_after_attention(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        residual_after_attention[i] = input[i] + attn_output[i];
    }

    const std::vector<float> ffn_input = RmsNorm(
        residual_after_attention, ffn_norm_weight, seq_len, hidden_size, rms_eps);
    const std::vector<float> ffn_output = GatedFeedForward(
        ffn_input,
        gate_weight,
        up_weight,
        down_weight,
        seq_len,
        hidden_size,
        intermediate_size);

    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = residual_after_attention[i] + ffn_output[i];
    }
    return output;
}

float MeanCrossEntropyLoss(
    const std::vector<float>& logits,
    const std::vector<int>& labels,
    std::size_t rows,
    std::size_t vocab_size,
    int ignore_index)
{
    if (rows == 0 || vocab_size == 0) {
        throw std::invalid_argument("MeanCrossEntropyLoss dimensions must be greater than zero");
    }
    if (logits.size() != rows * vocab_size) {
        throw std::invalid_argument("MeanCrossEntropyLoss logits size does not match rows * vocab_size");
    }
    if (labels.size() != rows) {
        throw std::invalid_argument("MeanCrossEntropyLoss labels size must match rows");
    }

    double total_loss = 0.0;
    std::size_t contributing_labels = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        const int label = labels[row];
        if (label == ignore_index) {
            continue;
        }
        if (label < 0 || static_cast<std::size_t>(label) >= vocab_size) {
            throw std::invalid_argument("MeanCrossEntropyLoss label is outside vocabulary");
        }

        const std::size_t offset = row * vocab_size;
        double max_logit = static_cast<double>(logits[offset]);
        for (std::size_t col = 1; col < vocab_size; ++col) {
            max_logit = std::max(max_logit, static_cast<double>(logits[offset + col]));
        }

        double exp_sum = 0.0;
        for (std::size_t col = 0; col < vocab_size; ++col) {
            exp_sum += std::exp(static_cast<double>(logits[offset + col]) - max_logit);
        }

        total_loss += std::log(exp_sum) + max_logit -
                      static_cast<double>(logits[offset + static_cast<std::size_t>(label)]);
        ++contributing_labels;
    }

    if (contributing_labels == 0) {
        throw std::invalid_argument("MeanCrossEntropyLoss requires at least one non-ignored label");
    }
    return static_cast<float>(total_loss / static_cast<double>(contributing_labels));
}

} // namespace minmind
