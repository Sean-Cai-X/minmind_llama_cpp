#include <minmind/sampling.hpp>

#include <stdexcept>

namespace minmind {

int GreedySample(const std::vector<float>& logits)
{
    if (logits.empty()) {
        throw std::invalid_argument("logits must not be empty");
    }

    std::size_t best = 0;
    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return static_cast<int>(best);
}

} // namespace minmind
