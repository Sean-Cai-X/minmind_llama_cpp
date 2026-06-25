#pragma once

#include <minmind/model.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace minmind {

struct TrainSample {
    TokenIds input_ids;
    TokenIds labels;
};

std::vector<TrainSample> BuildNextTokenTrainSamples(
    const TokenIds& token_ids,
    std::size_t sequence_length,
    std::size_t max_samples);
std::vector<TrainSample> LoadJsonlTrainSamples(const std::string& path);
void SaveJsonlTrainSamples(const std::string& path, const std::vector<TrainSample>& samples);

} // namespace minmind
