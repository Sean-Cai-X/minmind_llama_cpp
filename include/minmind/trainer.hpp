#pragma once

#include <minmind/dataset.hpp>
#include <minmind/model.hpp>

#include <cstddef>
#include <vector>

namespace minmind {

struct TrainLoopResult {
    std::vector<float> loss_history;
};

TrainLoopResult TrainLmHeadNoCache(
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
    float learning_rate,
    std::size_t steps);

TrainLoopResult TrainLmHeadSamplesNoCache(
    const std::vector<TrainSample>& samples,
    CausalLmWeights& weights,
    std::size_t vocab_size,
    std::size_t hidden_size,
    std::size_t num_heads,
    std::size_t intermediate_size,
    float rms_eps,
    float rope_theta,
    int ignore_index,
    float learning_rate,
    std::size_t epochs);

} // namespace minmind
