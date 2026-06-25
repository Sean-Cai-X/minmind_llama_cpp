#include "test_assert.hpp"

#include <minmind/config.hpp>

namespace minmind::tests {

void ConfigDefaultsAreValid()
{
    MiniMindConfig config;
    std::string error;
    MINMIND_REQUIRE(IsValid(config, &error));
    MINMIND_REQUIRE(error.empty());
}

void InvalidConfigReportsError()
{
    MiniMindConfig config;
    config.hidden_size = 65;
    std::string error;
    MINMIND_REQUIRE(!IsValid(config, &error));
    MINMIND_REQUIRE(!error.empty());
}

} // namespace minmind::tests
