#include "test_assert.hpp"

#include <minmind/model.hpp>
#include <minmind/sampling.hpp>

namespace minmind::tests {

void PlaceholderForwardChoosesNextToken()
{
    MiniMindConfig config;
    config.vocab_size = 16;
    MiniMindModel model(config);
    const std::vector<float> logits = model.ForwardLastToken(TokenIds{1, 2, 3});
    MINMIND_REQUIRE(logits.size() == 16);
    MINMIND_REQUIRE(GreedySample(logits) == 4);
}

} // namespace minmind::tests
