#include <minmind/trainer.hpp>

#include <stdexcept>

namespace minmind {

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
    std::size_t steps)
{
    if (steps == 0) {
        throw std::invalid_argument("training loop steps must be greater than zero");
    }

    TrainLoopResult result;
    result.loss_history.reserve(steps + 1);
    for (std::size_t step = 0; step < steps; ++step) {
        const TrainStepResult step_result = CausalLmTrainLmHeadOneStepNoCache(
            input_ids,
            labels,
            weights,
            vocab_size,
            hidden_size,
            num_heads,
            intermediate_size,
            rms_eps,
            rope_theta,
            ignore_index,
            learning_rate);

        if (step == 0) {
            result.loss_history.push_back(step_result.loss_before);
        }
        result.loss_history.push_back(step_result.loss_after);
    }
    return result;
}

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
    std::size_t epochs)
{
    if (samples.empty()) {
        throw std::invalid_argument("training samples must not be empty");
    }
    if (epochs == 0) {
        throw std::invalid_argument("training epochs must be greater than zero");
    }

    TrainLoopResult result;
    result.loss_history.reserve(epochs * samples.size() + 1);
    bool first_step = true;
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        for (const TrainSample& sample : samples) {
            const TrainStepResult step_result = CausalLmTrainLmHeadOneStepNoCache(
                sample.input_ids,
                sample.labels,
                weights,
                vocab_size,
                hidden_size,
                num_heads,
                intermediate_size,
                rms_eps,
                rope_theta,
                ignore_index,
                learning_rate);

            if (first_step) {
                result.loss_history.push_back(step_result.loss_before);
                first_step = false;
            }
            result.loss_history.push_back(step_result.loss_after);
        }
    }
    return result;
}

} // namespace minmind
