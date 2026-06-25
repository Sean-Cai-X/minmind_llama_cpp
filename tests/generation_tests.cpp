#include "test_assert.hpp"

#include <minmind/sampling.hpp>

namespace minmind::tests {

void GreedySampleChoosesLargestLogit()
{
    MINMIND_REQUIRE(GreedySample({0.0f, 1.0f, 0.5f}) == 1);
}

} // namespace minmind::tests
