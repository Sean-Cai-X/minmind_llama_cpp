#pragma once

#include <minmind/model.hpp>

#include <cstddef>
#include <string>

namespace minmind {

void SaveLmHeadCheckpoint(
    const std::string& path,
    const CausalLmWeights& weights,
    std::size_t vocab_size,
    std::size_t hidden_size);

void LoadLmHeadCheckpoint(
    const std::string& path,
    CausalLmWeights& weights,
    std::size_t vocab_size,
    std::size_t hidden_size);

} // namespace minmind
