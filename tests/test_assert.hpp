#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

#define MINMIND_REQUIRE(condition) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error(std::string("requirement failed: ") + #condition); \
        } \
    } while (false)

namespace minmind::tests {

using TestFn = void (*)();

int RunTest(const char* name, TestFn test);

} // namespace minmind::tests
