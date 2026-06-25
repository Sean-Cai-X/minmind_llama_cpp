#include "test_assert.hpp"

#include <fstream>
#include <sstream>
#include <string>

namespace minmind::tests {

namespace {

std::string ReadTextFile(const std::string& path)
{
    std::ifstream input(path);
    MINMIND_REQUIRE(input.good());
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::string ToolBlock(const std::string& manifest, const std::string& tool_name)
{
    const std::string marker = "\"name\": \"" + tool_name + "\"";
    const std::size_t name_pos = manifest.find(marker);
    MINMIND_REQUIRE(name_pos != std::string::npos);

    const std::size_t block_start = manifest.rfind("    {", name_pos);
    MINMIND_REQUIRE(block_start != std::string::npos);

    std::size_t block_end = manifest.find("\n    {\n      \"name\"", name_pos + marker.size());
    if (block_end == std::string::npos) {
        block_end = manifest.find("\n  ]", name_pos + marker.size());
    }
    MINMIND_REQUIRE(block_end != std::string::npos);
    return manifest.substr(block_start, block_end - block_start);
}

void RequireContains(const std::string& text, const std::string& needle)
{
    MINMIND_REQUIRE(text.find(needle) != std::string::npos);
}

} // namespace

void McpToolManifestIncludesCheckpointTrainingTools()
{
    const std::string manifest = ReadTextFile("apps/minmind_mcp_tools.json");

    const std::string lm_train = ToolBlock(manifest, "minmind_train_lm");
    RequireContains(lm_train, "\"--checkpoint\"");
    RequireContains(lm_train, "\"{checkpoint}\"");

    const std::string kg_train = ToolBlock(manifest, "minmind_train_kg");
    RequireContains(kg_train, "\"train-kg\"");
    RequireContains(kg_train, "\"--checkpoint\"");
    RequireContains(kg_train, "\"{checkpoint}\"");

    const std::string rgcn_train = ToolBlock(manifest, "minmind_train_rgcn");
    RequireContains(rgcn_train, "\"train-rgcn\"");
    RequireContains(rgcn_train, "\"--data\"");
    RequireContains(rgcn_train, "\"--nodes\"");
    RequireContains(rgcn_train, "\"--relations\"");
    RequireContains(rgcn_train, "\"--checkpoint\"");
    RequireContains(rgcn_train, "\"{checkpoint}\"");

    const std::string rgcn_inspect = ToolBlock(manifest, "minmind_inspect_rgcn_checkpoint");
    RequireContains(rgcn_inspect, "\"inspect-rgcn-checkpoint\"");
    RequireContains(rgcn_inspect, "\"--checkpoint\"");
    RequireContains(rgcn_inspect, "\"--nodes\"");
    RequireContains(rgcn_inspect, "\"--relations\"");
}

} // namespace minmind::tests
