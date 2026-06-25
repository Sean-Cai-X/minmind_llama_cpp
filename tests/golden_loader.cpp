#include "golden_loader.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifndef MINMIND_SOURCE_DIR
#define MINMIND_SOURCE_DIR "."
#endif

namespace minmind::tests {
namespace {

std::string ReadFile(const std::string& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open fixture: " + path);
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::size_t FindKey(const std::string& text, const std::string& key)
{
    const std::string quoted_key = "\"" + key + "\"";
    const std::size_t pos = text.find(quoted_key);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing fixture key: " + key);
    }
    return pos + quoted_key.size();
}

bool HasKey(const std::string& text, const std::string& key)
{
    return text.find("\"" + key + "\"") != std::string::npos;
}

std::string ExtractString(const std::string& text, const std::string& key)
{
    std::size_t pos = FindKey(text, key);
    pos = text.find(':', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing ':' for fixture key: " + key);
    }
    pos = text.find('"', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing string value for fixture key: " + key);
    }
    const std::size_t end = text.find('"', pos + 1);
    if (end == std::string::npos) {
        throw std::runtime_error("unterminated string value for fixture key: " + key);
    }
    return text.substr(pos + 1, end - pos - 1);
}

TokenIds ExtractIntArray(const std::string& text, const std::string& key)
{
    std::size_t pos = FindKey(text, key);
    pos = text.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing array value for fixture key: " + key);
    }
    const std::size_t end = text.find(']', pos);
    if (end == std::string::npos) {
        throw std::runtime_error("unterminated array value for fixture key: " + key);
    }

    TokenIds values;
    const std::string body = text.substr(pos + 1, end - pos - 1);
    std::size_t i = 0;
    while (i < body.size()) {
        while (i < body.size() && (std::isspace(static_cast<unsigned char>(body[i])) || body[i] == ',')) {
            ++i;
        }
        if (i >= body.size()) {
            break;
        }

        std::size_t token_end = i;
        if (body[token_end] == '-') {
            ++token_end;
        }
        while (token_end < body.size() && std::isdigit(static_cast<unsigned char>(body[token_end]))) {
            ++token_end;
        }
        if (token_end == i || (body[i] == '-' && token_end == i + 1)) {
            throw std::runtime_error("invalid integer in fixture key: " + key);
        }
        values.push_back(std::stoi(body.substr(i, token_end - i)));
        i = token_end;
    }
    return values;
}

double ExtractNumber(const std::string& text, const std::string& key)
{
    std::size_t pos = FindKey(text, key);
    pos = text.find(':', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing ':' for fixture key: " + key);
    }
    ++pos;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }

    const char* begin = text.c_str() + pos;
    char* end = nullptr;
    const double value = std::strtod(begin, &end);
    if (begin == end) {
        throw std::runtime_error("invalid number for fixture key: " + key);
    }
    return value;
}

std::size_t ExtractSize(const std::string& text, const std::string& key)
{
    const double value = ExtractNumber(text, key);
    if (value < 0.0) {
        throw std::runtime_error("negative size for fixture key: " + key);
    }
    return static_cast<std::size_t>(value);
}

std::vector<float> ExtractFloatArray(const std::string& text, const std::string& key)
{
    std::size_t pos = FindKey(text, key);
    pos = text.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing array value for fixture key: " + key);
    }
    const std::size_t end = text.find(']', pos);
    if (end == std::string::npos) {
        throw std::runtime_error("unterminated array value for fixture key: " + key);
    }

    std::vector<float> values;
    const std::string body = text.substr(pos + 1, end - pos - 1);
    std::size_t i = 0;
    while (i < body.size()) {
        while (i < body.size() && (std::isspace(static_cast<unsigned char>(body[i])) || body[i] == ',')) {
            ++i;
        }
        if (i >= body.size()) {
            break;
        }

        const char* begin = body.c_str() + i;
        char* number_end = nullptr;
        const double value = std::strtod(begin, &number_end);
        if (begin == number_end) {
            throw std::runtime_error("invalid float in fixture key: " + key);
        }
        values.push_back(static_cast<float>(value));
        i = static_cast<std::size_t>(number_end - body.c_str());
    }
    return values;
}

} // namespace

GoldenFixture LoadGoldenFixture(const std::string& relative_path)
{
    const std::string path = std::string(MINMIND_SOURCE_DIR) + "/" + relative_path;
    const std::string text = ReadFile(path);

    GoldenFixture fixture;
    fixture.name = ExtractString(text, "name");
    if (HasKey(text, "prompt")) {
        fixture.prompt = ExtractString(text, "prompt");
    }
    if (HasKey(text, "input_ids")) {
        fixture.input_ids = ExtractIntArray(text, "input_ids");
    }
    if (HasKey(text, "labels")) {
        fixture.labels = ExtractIntArray(text, "labels");
    }
    if (HasKey(text, "attention_mask")) {
        fixture.attention_mask = ExtractIntArray(text, "attention_mask");
    }
    if (HasKey(text, "greedy_ids")) {
        fixture.greedy_ids = ExtractIntArray(text, "greedy_ids");
    }
    if (HasKey(text, "loss")) {
        fixture.loss = ExtractNumber(text, "loss");
    }
    if (HasKey(text, "eps")) {
        fixture.eps = ExtractNumber(text, "eps");
    }
    if (HasKey(text, "rope_theta")) {
        fixture.rope_theta = ExtractNumber(text, "rope_theta");
    }
    if (HasKey(text, "rows")) {
        fixture.rows = ExtractSize(text, "rows");
    }
    if (HasKey(text, "cols")) {
        fixture.cols = ExtractSize(text, "cols");
    }
    if (HasKey(text, "seq_len")) {
        fixture.seq_len = ExtractSize(text, "seq_len");
    }
    if (HasKey(text, "vocab_size")) {
        fixture.vocab_size = ExtractSize(text, "vocab_size");
    }
    if (HasKey(text, "hidden_size")) {
        fixture.hidden_size = ExtractSize(text, "hidden_size");
    }
    if (HasKey(text, "intermediate_size")) {
        fixture.intermediate_size = ExtractSize(text, "intermediate_size");
    }
    if (HasKey(text, "num_heads")) {
        fixture.num_heads = ExtractSize(text, "num_heads");
    }
    if (HasKey(text, "logits_last")) {
        fixture.logits_last = ExtractFloatArray(text, "logits_last");
    }
    if (HasKey(text, "rmsnorm_input")) {
        fixture.rmsnorm_input = ExtractFloatArray(text, "rmsnorm_input");
    }
    if (HasKey(text, "rmsnorm_weight")) {
        fixture.rmsnorm_weight = ExtractFloatArray(text, "rmsnorm_weight");
    }
    if (HasKey(text, "rmsnorm_output")) {
        fixture.rmsnorm_output = ExtractFloatArray(text, "rmsnorm_output");
    }
    if (HasKey(text, "rope_input")) {
        fixture.rope_input = ExtractFloatArray(text, "rope_input");
    }
    if (HasKey(text, "rope_cos")) {
        fixture.rope_cos = ExtractFloatArray(text, "rope_cos");
    }
    if (HasKey(text, "rope_sin")) {
        fixture.rope_sin = ExtractFloatArray(text, "rope_sin");
    }
    if (HasKey(text, "rope_output")) {
        fixture.rope_output = ExtractFloatArray(text, "rope_output");
    }
    if (HasKey(text, "ffn_input")) {
        fixture.ffn_input = ExtractFloatArray(text, "ffn_input");
    }
    if (HasKey(text, "gate_weight")) {
        fixture.gate_weight = ExtractFloatArray(text, "gate_weight");
    }
    if (HasKey(text, "up_weight")) {
        fixture.up_weight = ExtractFloatArray(text, "up_weight");
    }
    if (HasKey(text, "down_weight")) {
        fixture.down_weight = ExtractFloatArray(text, "down_weight");
    }
    if (HasKey(text, "gate_output")) {
        fixture.gate_output = ExtractFloatArray(text, "gate_output");
    }
    if (HasKey(text, "up_output")) {
        fixture.up_output = ExtractFloatArray(text, "up_output");
    }
    if (HasKey(text, "ffn_output")) {
        fixture.ffn_output = ExtractFloatArray(text, "ffn_output");
    }
    if (HasKey(text, "attention_input")) {
        fixture.attention_input = ExtractFloatArray(text, "attention_input");
    }
    if (HasKey(text, "q_weight")) {
        fixture.q_weight = ExtractFloatArray(text, "q_weight");
    }
    if (HasKey(text, "k_weight")) {
        fixture.k_weight = ExtractFloatArray(text, "k_weight");
    }
    if (HasKey(text, "v_weight")) {
        fixture.v_weight = ExtractFloatArray(text, "v_weight");
    }
    if (HasKey(text, "o_weight")) {
        fixture.o_weight = ExtractFloatArray(text, "o_weight");
    }
    if (HasKey(text, "attention_output")) {
        fixture.attention_output = ExtractFloatArray(text, "attention_output");
    }
    if (HasKey(text, "block_input")) {
        fixture.block_input = ExtractFloatArray(text, "block_input");
    }
    if (HasKey(text, "attn_norm_weight")) {
        fixture.attn_norm_weight = ExtractFloatArray(text, "attn_norm_weight");
    }
    if (HasKey(text, "ffn_norm_weight")) {
        fixture.ffn_norm_weight = ExtractFloatArray(text, "ffn_norm_weight");
    }
    if (HasKey(text, "final_norm_weight")) {
        fixture.final_norm_weight = ExtractFloatArray(text, "final_norm_weight");
    }
    if (HasKey(text, "block_attention_output")) {
        fixture.block_attention_output = ExtractFloatArray(text, "block_attention_output");
    }
    if (HasKey(text, "block_output")) {
        fixture.block_output = ExtractFloatArray(text, "block_output");
    }
    if (HasKey(text, "token_embedding")) {
        fixture.token_embedding = ExtractFloatArray(text, "token_embedding");
    }
    if (HasKey(text, "lm_head_weight")) {
        fixture.lm_head_weight = ExtractFloatArray(text, "lm_head_weight");
    }
    if (HasKey(text, "logits")) {
        fixture.logits = ExtractFloatArray(text, "logits");
    }
    return fixture;
}

} // namespace minmind::tests
