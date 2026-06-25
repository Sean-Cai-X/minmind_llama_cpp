#include <minmind/checkpoint.hpp>
#include <minmind/dataset.hpp>
#include <minmind/kg.hpp>
#include <minmind/model.hpp>
#include <minmind/sampling.hpp>
#include <minmind/tokenizer.hpp>
#include <minmind/trainer.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kTinyVocabSize = 4;
constexpr std::size_t kTinyHiddenSize = 4;
constexpr std::size_t kTinyNumHeads = 1;
constexpr std::size_t kTinyIntermediateSize = 4;
constexpr float kTinyRmsEps = 0.000001f;
constexpr float kTinyRopeTheta = 1000000.0f;
constexpr int kIgnoreIndex = -100;
constexpr const char* kTokenizerVocab4 = "vocab4";
constexpr const char* kTokenizerByte = "byte";

struct TrainCommandOptions {
    std::string config_path;
    std::string profile_path;
    std::string profile_name;
    std::string data_path;
    std::string checkpoint_path;
    std::size_t vocab_size = kTinyVocabSize;
    std::size_t epochs = 4;
    float learning_rate = 0.25f;
    bool json_output = false;
};

struct PrepareTextCommandOptions {
    std::string config_path;
    std::string input_path;
    std::string output_path;
    std::string tokenizer = kTokenizerVocab4;
    std::string vocab_input_path;
    std::string vocab_output_path;
    std::size_t vocab_size = kTinyVocabSize;
    std::size_t sequence_length = 16;
    std::size_t max_samples = 0;
    bool json_output = false;
};

struct PipelineCommandOptions {
    std::string config_path;
    std::string profile_path;
    std::string profile_name;
    PrepareTextCommandOptions prepare;
    TrainCommandOptions train;
    bool json_output = false;
};

struct InspectDatasetCommandOptions {
    std::string data_path;
    std::size_t vocab_size = kTinyVocabSize;
    bool json_output = false;
};

struct InspectTextCommandOptions {
    std::string input_path;
    std::string tokenizer = kTokenizerVocab4;
    std::string vocab_input_path;
    std::size_t vocab_size = kTinyVocabSize;
    bool json_output = false;
};

struct InspectKgDatasetCommandOptions {
    std::string data_path;
    std::size_t entity_count = 0;
    std::size_t relation_count = 0;
    bool json_output = false;
};

struct PrepareKgCommandOptions {
    std::string input_path;
    std::string output_path;
    std::string entity_map_path;
    std::string relation_map_path;
    std::size_t max_triples = 0;
    bool json_output = false;
};

struct InspectCheckpointCommandOptions {
    std::string checkpoint_path;
    std::size_t vocab_size = kTinyVocabSize;
    std::size_t hidden_size = kTinyHiddenSize;
    bool json_output = false;
};

struct GenerateCheckpointCommandOptions {
    std::string checkpoint_path;
    std::string input = "1";
    std::size_t vocab_size = kTinyVocabSize;
    std::size_t hidden_size = kTinyHiddenSize;
    std::size_t max_new_tokens = 1;
    int eos_token_id = -1;
    bool json_output = false;
};

struct ValidateLmProfileCommandOptions {
    std::string profile_path;
    std::string profile_name;
    std::string input = "0,1";
    std::size_t max_new_tokens = 2;
    bool json_output = false;
};

struct ValidateKgProfileCommandOptions {
    std::string profile_path;
    std::string profile_name;
    std::string model_name = "transe";
    bool json_output = false;
};

struct ValidateRgcnProfileCommandOptions {
    std::string profile_path;
    std::string profile_name;
    bool json_output = false;
};

struct TrainKgCommandOptions {
    std::string profile_path;
    std::string profile_name;
    std::string data_path;
    std::string checkpoint_path;
    std::string model_name = "transe";
    std::size_t entity_count = 0;
    std::size_t relation_count = 0;
    std::size_t embedding_dim = 8;
    std::size_t epochs = 4;
    std::size_t max_triples = 0;
    float learning_rate = 0.01f;
    float margin = 1.0f;
    bool json_output = false;
};

struct InspectKgCheckpointCommandOptions {
    std::string checkpoint_path;
    std::string model_name = "transe";
    std::size_t entity_count = 0;
    std::size_t relation_count = 0;
    std::size_t embedding_dim = 0;
    bool json_output = false;
};

struct KgPipelineCommandOptions {
    TrainKgCommandOptions train;
    bool json_output = false;
};

struct TrainRgcnCommandOptions {
    std::string profile_path;
    std::string profile_name;
    std::string data_path;
    std::string checkpoint_path;
    std::size_t node_count = 0;
    std::size_t relation_count = 0;
    std::size_t feature_dim = 8;
    std::size_t epochs = 4;
    std::size_t max_triples = 0;
    float learning_rate = 0.01f;
    bool json_output = false;
};

struct PrepareKgPipelineCommandOptions {
    std::string profile_path;
    std::string profile_name;
    std::string target = "kg";
    std::string model_name = "transe";
    PrepareKgCommandOptions prepare;
    TrainKgCommandOptions train_kg;
    TrainRgcnCommandOptions train_rgcn;
    bool json_output = false;
};

struct InspectRgcnCheckpointCommandOptions {
    std::string checkpoint_path;
    std::size_t node_count = 0;
    std::size_t relation_count = 0;
    std::size_t feature_dim = 0;
    bool json_output = false;
};

struct RgcnPipelineCommandOptions {
    TrainRgcnCommandOptions train;
    bool json_output = false;
};

std::vector<float> Identity(std::size_t size)
{
    std::vector<float> values(size * size, 0.0f);
    for (std::size_t i = 0; i < size; ++i) {
        values[i * size + i] = 1.0f;
    }
    return values;
}

minmind::CausalLmWeights MakeTinyWeights(std::size_t vocab_size)
{
    const std::vector<float> identity = Identity(kTinyHiddenSize);

    minmind::CausalLmWeights weights;
    weights.token_embedding.resize(vocab_size * kTinyHiddenSize, 0.0f);
    for (std::size_t token = 0; token < vocab_size; ++token) {
        const float token_scale = static_cast<float>((token % kTinyHiddenSize) + 1);
        for (std::size_t dim = 0; dim < kTinyHiddenSize; ++dim) {
            weights.token_embedding[token * kTinyHiddenSize + dim] =
                dim == (token % kTinyHiddenSize) ? token_scale : 0.0f;
        }
    }
    weights.attn_norm_weight = {1.0f, 1.0f, 1.0f, 1.0f};
    weights.ffn_norm_weight = {1.0f, 1.0f, 1.0f, 1.0f};
    weights.final_norm_weight = {1.0f, 1.0f, 1.0f, 1.0f};
    weights.q_weight = identity;
    weights.k_weight = identity;
    weights.v_weight = identity;
    weights.o_weight = identity;
    weights.gate_weight = identity;
    weights.up_weight = identity;
    weights.down_weight = identity;
    weights.lm_head_weight.resize(vocab_size * kTinyHiddenSize, 0.0f);
    for (std::size_t token = 0; token < vocab_size; ++token) {
        weights.lm_head_weight[token * kTinyHiddenSize + (token % kTinyHiddenSize)] = 1.0f;
    }
    return weights;
}

void PrintUsage()
{
    std::cout
        << "usage:\n"
        << "  minmind_cli [token_ids]\n"
        << "  minmind_cli generate [token_ids]\n"
        << "  minmind_cli prepare-text [--config <json>] --input <txt> --output <jsonl> [--tokenizer vocab4|byte]\n"
        << "      [--vocab-size N] [--vocab-in <vocab>] [--vocab-out <vocab>] [--seq-len N] [--max-samples N] [--json]\n"
        << "  minmind_cli train [--config <json>] [--profile <json> --profile-name <name>]\n"
        << "      --data <jsonl> --checkpoint <path>\n"
        << "      [--vocab-size N] [--epochs N] [--learning-rate F] [--json]\n"
        << "  minmind_cli inspect-dataset --data <jsonl> [--vocab-size N] [--json]\n"
        << "  minmind_cli inspect-text --input <txt> [--tokenizer vocab4|byte]\n"
        << "      [--vocab-size N] [--vocab-in <vocab>] [--json]\n"
        << "  minmind_cli prepare-kg --input <tsv> --output <tsv> [--entity-map <path>]\n"
        << "      [--relation-map <path>] [--max-triples N] [--json]\n"
        << "  minmind_cli prepare-kg-pipeline --profile <json> --profile-name <name>\n"
        << "      [--target kg|rgcn] [--model transe|distmult|rotate] [--json]\n"
        << "  minmind_cli inspect-kg-dataset --data <tsv> [--entities N] [--relations N] [--json]\n"
        << "  minmind_cli inspect-checkpoint --checkpoint <path> [--vocab-size N] [--hidden-size N] [--json]\n"
        << "  minmind_cli generate-checkpoint --checkpoint <path> --input <token_ids>\n"
        << "      [--vocab-size N] [--hidden-size N] [--max-new N] [--eos N] [--json]\n"
        << "  minmind_cli validate-lm-profile --profile <json> --profile-name <name> [--input <token_ids>] [--max-new N] [--json]\n"
        << "  minmind_cli train-kg [--profile <json> --profile-name <name>] --data <tsv>\n"
        << "      --model transe|distmult|rotate --entities N --relations N\n"
        << "      [--dim N] [--epochs N] [--max-triples N] [--learning-rate F] [--margin F] [--checkpoint <path>] [--json]\n"
        << "  minmind_cli inspect-kg-checkpoint --checkpoint <path> --model transe|distmult|rotate\n"
        << "      --entities N --relations N --dim N [--json]\n"
        << "  minmind_cli kg-pipeline --profile <json> --profile-name <name> --model transe|distmult|rotate [--json]\n"
        << "  minmind_cli validate-kg-profile --profile <json> --profile-name <name> --model transe|distmult|rotate [--json]\n"
        << "  minmind_cli train-rgcn [--profile <json> --profile-name <name>] --data <tsv>\n"
        << "      --nodes N --relations N [--dim N] [--epochs N] [--max-triples N] [--learning-rate F] [--checkpoint <path>] [--json]\n"
        << "  minmind_cli inspect-rgcn-checkpoint --checkpoint <path> --nodes N --relations N --dim N [--json]\n"
        << "  minmind_cli rgcn-pipeline --profile <json> --profile-name <name> [--json]\n"
        << "  minmind_cli validate-rgcn-profile --profile <json> --profile-name <name> [--json]\n"
        << "  minmind_cli pipeline [--config <json>] [--profile <json> --profile-name <name>] [--json]\n";
}

std::string JsonEscape(const std::string& value)
{
    std::string escaped;
    escaped.reserve(value.size() + 2);
    for (const char ch : value) {
        switch (ch) {
        case '"':
            escaped += "\\\"";
            break;
        case '\\':
            escaped += "\\\\";
            break;
        case '\b':
            escaped += "\\b";
            break;
        case '\f':
            escaped += "\\f";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                std::ostringstream out;
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(static_cast<unsigned char>(ch));
                escaped += out.str();
            } else {
                escaped.push_back(ch);
            }
            break;
        }
    }
    return escaped;
}

void WriteJsonStringField(std::ostream& out, const std::string& name, const std::string& value)
{
    out << '"' << name << "\":\"" << JsonEscape(value) << '"';
}

std::size_t ParseSizeT(const std::string& value, const std::string& name)
{
    std::size_t parsed = 0;
    const unsigned long long result = std::stoull(value, &parsed);
    if (parsed != value.size()) {
        throw std::invalid_argument("invalid integer for " + name + ": " + value);
    }
    return static_cast<std::size_t>(result);
}

std::string ReadTextFile(const std::string& path);

float ParseFloat(const std::string& value, const std::string& name)
{
    std::size_t parsed = 0;
    const float result = std::stof(value, &parsed);
    if (parsed != value.size()) {
        throw std::invalid_argument("invalid float for " + name + ": " + value);
    }
    return result;
}

std::string ReadOptionValue(int argc, char** argv, int& index, const std::string& name)
{
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + name);
    }
    ++index;
    return argv[index];
}

void EnsureParentDirectory(const std::string& path)
{
    const std::filesystem::path output_path(path);
    const std::filesystem::path parent = output_path.parent_path();
    if (parent.empty()) {
        return;
    }

    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error) {
        throw std::runtime_error("failed to create parent directory for " + path + ": " + error.message());
    }
}

std::size_t FindJsonValue(const std::string& text, const std::string& key)
{
    const std::string quoted_key = "\"" + key + "\"";
    const std::size_t key_pos = text.find(quoted_key);
    if (key_pos == std::string::npos) {
        return std::string::npos;
    }

    const std::size_t colon_pos = text.find(':', key_pos + quoted_key.size());
    if (colon_pos == std::string::npos) {
        throw std::invalid_argument("invalid config key without value: " + key);
    }

    std::size_t value_pos = colon_pos + 1;
    while (value_pos < text.size() && std::isspace(static_cast<unsigned char>(text[value_pos]))) {
        ++value_pos;
    }
    return value_pos;
}

bool TryReadJsonString(const std::string& text, const std::string& key, std::string* value)
{
    std::size_t pos = FindJsonValue(text, key);
    if (pos == std::string::npos) {
        return false;
    }
    if (pos >= text.size() || text[pos] != '"') {
        throw std::invalid_argument("config key must be a string: " + key);
    }
    ++pos;

    std::string parsed;
    while (pos < text.size()) {
        const char ch = text[pos++];
        if (ch == '"') {
            *value = std::move(parsed);
            return true;
        }
        if (ch == '\\') {
            if (pos >= text.size()) {
                throw std::invalid_argument("unterminated escape in config key: " + key);
            }
            const char escaped = text[pos++];
            if (escaped == '"' || escaped == '\\' || escaped == '/') {
                parsed.push_back(escaped);
            } else if (escaped == 'n') {
                parsed.push_back('\n');
            } else if (escaped == 'r') {
                parsed.push_back('\r');
            } else if (escaped == 't') {
                parsed.push_back('\t');
            } else {
                throw std::invalid_argument("unsupported escape in config key: " + key);
            }
        } else {
            parsed.push_back(ch);
        }
    }

    throw std::invalid_argument("unterminated string in config key: " + key);
}

bool TryReadJsonNumber(const std::string& text, const std::string& key, std::string* value)
{
    const std::size_t pos = FindJsonValue(text, key);
    if (pos == std::string::npos) {
        return false;
    }

    std::size_t end = pos;
    while (end < text.size()) {
        const char ch = text[end];
        if (!std::isdigit(static_cast<unsigned char>(ch)) &&
            ch != '-' &&
            ch != '+' &&
            ch != '.' &&
            ch != 'e' &&
            ch != 'E') {
            break;
        }
        ++end;
    }
    if (end == pos) {
        throw std::invalid_argument("config key must be a number: " + key);
    }

    *value = text.substr(pos, end - pos);
    return true;
}

std::string ExtractJsonObject(const std::string& text, const std::string& key)
{
    const std::size_t value_pos = FindJsonValue(text, key);
    if (value_pos == std::string::npos) {
        return {};
    }
    if (value_pos >= text.size() || text[value_pos] != '{') {
        throw std::invalid_argument("config key must be an object: " + key);
    }

    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t pos = value_pos; pos < text.size(); ++pos) {
        const char ch = text[pos];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
        } else if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0) {
                return text.substr(value_pos, pos - value_pos + 1);
            }
        }
    }
    throw std::invalid_argument("unterminated config object: " + key);
}

void ApplyTrainKgProfile(const std::string& profile_path, const std::string& profile_name, TrainKgCommandOptions* options)
{
    const std::string text = ReadTextFile(profile_path);
    const std::string profiles = ExtractJsonObject(text, "profiles");
    if (profiles.empty()) {
        throw std::invalid_argument("training profile file requires profiles object");
    }
    const std::string profile = ExtractJsonObject(profiles, profile_name);
    if (profile.empty()) {
        throw std::invalid_argument("training profile not found: " + profile_name);
    }

    std::string value;
    if (TryReadJsonString(profile, "data", &value) || TryReadJsonString(profile, "output", &value)) {
        options->data_path = value;
    }
    if (TryReadJsonString(profile, "checkpoint", &value)) {
        options->checkpoint_path = value;
    }
    if (TryReadJsonString(profile, "checkpoint", &value)) {
        options->checkpoint_path = value;
    }
    if (TryReadJsonNumber(profile, "entities", &value) || TryReadJsonNumber(profile, "max_entities", &value)) {
        options->entity_count = ParseSizeT(value, "entities");
    }
    if (TryReadJsonNumber(profile, "relations", &value) || TryReadJsonNumber(profile, "max_relations", &value)) {
        options->relation_count = ParseSizeT(value, "relations");
    }
    if (TryReadJsonNumber(profile, "dim", &value)) {
        options->embedding_dim = ParseSizeT(value, "dim");
    }
    if (TryReadJsonNumber(profile, "epochs", &value)) {
        options->epochs = ParseSizeT(value, "epochs");
    }
    if (TryReadJsonNumber(profile, "max_triples", &value)) {
        options->max_triples = ParseSizeT(value, "max_triples");
    }
    if (TryReadJsonNumber(profile, "learning_rate", &value)) {
        options->learning_rate = ParseFloat(value, "learning_rate");
    }
    if (TryReadJsonNumber(profile, "margin", &value)) {
        options->margin = ParseFloat(value, "margin");
    }
}

void ApplyTrainRgcnProfile(const std::string& profile_path, const std::string& profile_name, TrainRgcnCommandOptions* options)
{
    const std::string text = ReadTextFile(profile_path);
    const std::string profiles = ExtractJsonObject(text, "profiles");
    if (profiles.empty()) {
        throw std::invalid_argument("training profile file requires profiles object");
    }
    const std::string profile = ExtractJsonObject(profiles, profile_name);
    if (profile.empty()) {
        throw std::invalid_argument("training profile not found: " + profile_name);
    }

    std::string value;
    if (TryReadJsonString(profile, "data", &value) || TryReadJsonString(profile, "output", &value)) {
        options->data_path = value;
    }
    if (TryReadJsonNumber(profile, "nodes", &value) ||
        TryReadJsonNumber(profile, "node_count", &value) ||
        TryReadJsonNumber(profile, "entities", &value)) {
        options->node_count = ParseSizeT(value, "nodes");
    }
    if (TryReadJsonNumber(profile, "relations", &value) || TryReadJsonNumber(profile, "relation_count", &value)) {
        options->relation_count = ParseSizeT(value, "relations");
    }
    if (TryReadJsonNumber(profile, "dim", &value) || TryReadJsonNumber(profile, "feature_dim", &value)) {
        options->feature_dim = ParseSizeT(value, "dim");
    }
    if (TryReadJsonNumber(profile, "epochs", &value)) {
        options->epochs = ParseSizeT(value, "epochs");
    }
    if (TryReadJsonNumber(profile, "max_triples", &value)) {
        options->max_triples = ParseSizeT(value, "max_triples");
    }
    if (TryReadJsonNumber(profile, "learning_rate", &value)) {
        options->learning_rate = ParseFloat(value, "learning_rate");
    }
}

void ApplyPrepareKgPipelineProfile(
    const std::string& profile_path,
    const std::string& profile_name,
    PrepareKgPipelineCommandOptions* options)
{
    const std::string text = ReadTextFile(profile_path);
    const std::string profiles = ExtractJsonObject(text, "profiles");
    if (profiles.empty()) {
        throw std::invalid_argument("training profile file requires profiles object");
    }
    const std::string profile = ExtractJsonObject(profiles, profile_name);
    if (profile.empty()) {
        throw std::invalid_argument("training profile not found: " + profile_name);
    }

    std::string value;
    if (TryReadJsonString(profile, "input", &value)) {
        options->prepare.input_path = value;
    }
    if (TryReadJsonString(profile, "output", &value) || TryReadJsonString(profile, "data", &value)) {
        options->prepare.output_path = value;
        options->train_kg.data_path = value;
        options->train_rgcn.data_path = value;
    }
    if (TryReadJsonString(profile, "entity_map", &value)) {
        options->prepare.entity_map_path = value;
    }
    if (TryReadJsonString(profile, "relation_map", &value)) {
        options->prepare.relation_map_path = value;
    }
    if (TryReadJsonString(profile, "checkpoint", &value)) {
        options->train_kg.checkpoint_path = value;
        options->train_rgcn.checkpoint_path = value;
    }
    if (TryReadJsonNumber(profile, "max_triples", &value)) {
        const std::size_t max_triples = ParseSizeT(value, "max_triples");
        options->prepare.max_triples = max_triples;
        options->train_kg.max_triples = max_triples;
        options->train_rgcn.max_triples = max_triples;
    }
    if (TryReadJsonNumber(profile, "entities", &value) || TryReadJsonNumber(profile, "nodes", &value)) {
        const std::size_t count = ParseSizeT(value, "entities");
        options->train_kg.entity_count = count;
        options->train_rgcn.node_count = count;
    }
    if (TryReadJsonNumber(profile, "relations", &value)) {
        const std::size_t count = ParseSizeT(value, "relations");
        options->train_kg.relation_count = count;
        options->train_rgcn.relation_count = count;
    }
    if (TryReadJsonNumber(profile, "dim", &value)) {
        const std::size_t dim = ParseSizeT(value, "dim");
        options->train_kg.embedding_dim = dim;
        options->train_rgcn.feature_dim = dim;
    }
    if (TryReadJsonNumber(profile, "epochs", &value)) {
        const std::size_t epochs = ParseSizeT(value, "epochs");
        options->train_kg.epochs = epochs;
        options->train_rgcn.epochs = epochs;
    }
    if (TryReadJsonNumber(profile, "learning_rate", &value)) {
        const float learning_rate = ParseFloat(value, "learning_rate");
        options->train_kg.learning_rate = learning_rate;
        options->train_rgcn.learning_rate = learning_rate;
    }

    options->train_kg.profile_path = profile_path;
    options->train_kg.profile_name = profile_name;
    options->train_rgcn.profile_path = profile_path;
    options->train_rgcn.profile_name = profile_name;
}

void ApplyTrainProfile(const std::string& profile_path, const std::string& profile_name, TrainCommandOptions* options)
{
    const std::string text = ReadTextFile(profile_path);
    const std::string profiles = ExtractJsonObject(text, "profiles");
    if (profiles.empty()) {
        throw std::invalid_argument("training profile file requires profiles object");
    }
    const std::string profile = ExtractJsonObject(profiles, profile_name);
    if (profile.empty()) {
        throw std::invalid_argument("training profile not found: " + profile_name);
    }

    std::string value;
    if (TryReadJsonString(profile, "data", &value)) {
        options->data_path = value;
    }
    if (TryReadJsonString(profile, "checkpoint", &value)) {
        options->checkpoint_path = value;
    }
    if (TryReadJsonNumber(profile, "vocab_size", &value)) {
        options->vocab_size = ParseSizeT(value, "vocab_size");
    }
    if (TryReadJsonNumber(profile, "epochs", &value)) {
        options->epochs = ParseSizeT(value, "epochs");
    }
    if (TryReadJsonNumber(profile, "learning_rate", &value)) {
        options->learning_rate = ParseFloat(value, "learning_rate");
    }
}

void ApplyPipelineProfile(const std::string& profile_path, const std::string& profile_name, PipelineCommandOptions* options)
{
    const std::string text = ReadTextFile(profile_path);
    const std::string profiles = ExtractJsonObject(text, "profiles");
    if (profiles.empty()) {
        throw std::invalid_argument("training profile file requires profiles object");
    }
    const std::string profile = ExtractJsonObject(profiles, profile_name);
    if (profile.empty()) {
        throw std::invalid_argument("training profile not found: " + profile_name);
    }

    std::string value;
    if (TryReadJsonString(profile, "input", &value) || TryReadJsonString(profile, "prepare_input", &value)) {
        options->prepare.input_path = value;
    }
    if (TryReadJsonString(profile, "output", &value) || TryReadJsonString(profile, "prepare_output", &value)) {
        options->prepare.output_path = value;
    }
    if (TryReadJsonString(profile, "tokenizer", &value) || TryReadJsonString(profile, "prepare_tokenizer", &value)) {
        options->prepare.tokenizer = value;
    }
    if (TryReadJsonString(profile, "vocab_in", &value) || TryReadJsonString(profile, "prepare_vocab_in", &value)) {
        options->prepare.vocab_input_path = value;
    }
    if (TryReadJsonString(profile, "vocab_out", &value) || TryReadJsonString(profile, "prepare_vocab_out", &value)) {
        options->prepare.vocab_output_path = value;
    }
    if (TryReadJsonNumber(profile, "vocab_size", &value) || TryReadJsonNumber(profile, "prepare_vocab_size", &value)) {
        options->prepare.vocab_size = ParseSizeT(value, "vocab_size");
        options->train.vocab_size = options->prepare.vocab_size;
    }
    if (TryReadJsonNumber(profile, "sequence_length", &value) || TryReadJsonNumber(profile, "seq_len", &value)) {
        options->prepare.sequence_length = ParseSizeT(value, "sequence_length");
    }
    if (TryReadJsonNumber(profile, "max_samples", &value)) {
        options->prepare.max_samples = ParseSizeT(value, "max_samples");
    }
    if (TryReadJsonString(profile, "checkpoint", &value) || TryReadJsonString(profile, "train_checkpoint", &value)) {
        options->train.checkpoint_path = value;
    }
    if (TryReadJsonNumber(profile, "epochs", &value) || TryReadJsonNumber(profile, "train_epochs", &value)) {
        options->train.epochs = ParseSizeT(value, "epochs");
    }
    if (TryReadJsonNumber(profile, "learning_rate", &value) || TryReadJsonNumber(profile, "train_learning_rate", &value)) {
        options->train.learning_rate = ParseFloat(value, "learning_rate");
    }
    options->train.data_path = options->prepare.output_path;
    options->train.profile_path = profile_path;
    options->train.profile_name = profile_name;
}

void ApplyTrainConfig(const std::string& config_path, TrainCommandOptions* options)
{
    const std::string text = ReadTextFile(config_path);
    std::string value;
    if (TryReadJsonString(text, "data", &value) || TryReadJsonString(text, "data_path", &value)) {
        options->data_path = value;
    }
    if (TryReadJsonString(text, "checkpoint", &value) || TryReadJsonString(text, "checkpoint_path", &value)) {
        options->checkpoint_path = value;
    }
    if (TryReadJsonNumber(text, "vocab_size", &value)) {
        options->vocab_size = ParseSizeT(value, "vocab_size");
    }
    if (TryReadJsonNumber(text, "epochs", &value)) {
        options->epochs = ParseSizeT(value, "epochs");
    }
    if (TryReadJsonNumber(text, "learning_rate", &value)) {
        options->learning_rate = ParseFloat(value, "learning_rate");
    }
}

void ApplyPrepareTextConfig(const std::string& config_path, PrepareTextCommandOptions* options)
{
    const std::string text = ReadTextFile(config_path);
    std::string value;
    if (TryReadJsonString(text, "input", &value) || TryReadJsonString(text, "input_path", &value)) {
        options->input_path = value;
    }
    if (TryReadJsonString(text, "output", &value) || TryReadJsonString(text, "output_path", &value)) {
        options->output_path = value;
    }
    if (TryReadJsonString(text, "tokenizer", &value)) {
        options->tokenizer = value;
    }
    if (TryReadJsonString(text, "vocab_in", &value) || TryReadJsonString(text, "vocab_input_path", &value)) {
        options->vocab_input_path = value;
    }
    if (TryReadJsonString(text, "vocab_out", &value) || TryReadJsonString(text, "vocab_output_path", &value)) {
        options->vocab_output_path = value;
    }
    if (TryReadJsonNumber(text, "vocab_size", &value)) {
        options->vocab_size = ParseSizeT(value, "vocab_size");
    }
    if (TryReadJsonNumber(text, "seq_len", &value) || TryReadJsonNumber(text, "sequence_length", &value)) {
        options->sequence_length = ParseSizeT(value, "seq_len");
    }
    if (TryReadJsonNumber(text, "max_samples", &value)) {
        options->max_samples = ParseSizeT(value, "max_samples");
    }
}

void ApplyPipelineConfig(const std::string& config_path, PipelineCommandOptions* options)
{
    const std::string text = ReadTextFile(config_path);
    std::string value;

    if (TryReadJsonString(text, "prepare_config", &value)) {
        options->prepare.config_path = value;
        ApplyPrepareTextConfig(value, &options->prepare);
    }
    if (TryReadJsonString(text, "train_config", &value)) {
        options->train.config_path = value;
        ApplyTrainConfig(value, &options->train);
    }

    if (TryReadJsonString(text, "prepare_input", &value) || TryReadJsonString(text, "input", &value)) {
        options->prepare.input_path = value;
    }
    if (TryReadJsonString(text, "prepare_output", &value) || TryReadJsonString(text, "output", &value)) {
        options->prepare.output_path = value;
    }
    if (TryReadJsonString(text, "prepare_tokenizer", &value) || TryReadJsonString(text, "tokenizer", &value)) {
        options->prepare.tokenizer = value;
    }
    if (TryReadJsonString(text, "prepare_vocab_in", &value) || TryReadJsonString(text, "vocab_in", &value)) {
        options->prepare.vocab_input_path = value;
    }
    if (TryReadJsonString(text, "prepare_vocab_out", &value) || TryReadJsonString(text, "vocab_out", &value)) {
        options->prepare.vocab_output_path = value;
    }
    if (TryReadJsonNumber(text, "prepare_vocab_size", &value) || TryReadJsonNumber(text, "vocab_size", &value)) {
        options->prepare.vocab_size = ParseSizeT(value, "prepare_vocab_size");
    }
    if (TryReadJsonNumber(text, "prepare_seq_len", &value) || TryReadJsonNumber(text, "seq_len", &value)) {
        options->prepare.sequence_length = ParseSizeT(value, "prepare_seq_len");
    }
    if (TryReadJsonNumber(text, "prepare_max_samples", &value) || TryReadJsonNumber(text, "max_samples", &value)) {
        options->prepare.max_samples = ParseSizeT(value, "prepare_max_samples");
    }

    if (TryReadJsonString(text, "train_data", &value) || TryReadJsonString(text, "data", &value)) {
        options->train.data_path = value;
    }
    if (TryReadJsonString(text, "train_checkpoint", &value) || TryReadJsonString(text, "checkpoint", &value)) {
        options->train.checkpoint_path = value;
    }
    if (TryReadJsonNumber(text, "train_vocab_size", &value)) {
        options->train.vocab_size = ParseSizeT(value, "train_vocab_size");
    }
    if (TryReadJsonNumber(text, "train_epochs", &value) || TryReadJsonNumber(text, "epochs", &value)) {
        options->train.epochs = ParseSizeT(value, "train_epochs");
    }
    if (TryReadJsonNumber(text, "train_learning_rate", &value) || TryReadJsonNumber(text, "learning_rate", &value)) {
        options->train.learning_rate = ParseFloat(value, "train_learning_rate");
    }

    if (options->train.data_path.empty()) {
        options->train.data_path = options->prepare.output_path;
    }
    if (options->train.vocab_size == kTinyVocabSize && options->prepare.tokenizer == kTokenizerByte) {
        options->train.vocab_size = options->prepare.vocab_size;
    }
}

TrainCommandOptions ParseTrainOptions(int argc, char** argv)
{
    TrainCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            options.profile_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--profile-name") {
            options.profile_name = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--config") {
            options.config_path = ReadOptionValue(argc, argv, i, arg);
        }
    }
    if (!options.profile_path.empty() || !options.profile_name.empty()) {
        if (options.profile_path.empty() || options.profile_name.empty()) {
            throw std::invalid_argument("train profile requires --profile and --profile-name");
        }
        ApplyTrainProfile(options.profile_path, options.profile_name, &options);
    }
    if (!options.config_path.empty()) {
        ApplyTrainConfig(options.config_path, &options);
    }

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile" || arg == "--profile-name" || arg == "--config") {
            ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--data") {
            options.data_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--checkpoint" || arg == "--out") {
            options.checkpoint_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--vocab-size") {
            options.vocab_size = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--epochs") {
            options.epochs = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--learning-rate") {
            options.learning_rate = ParseFloat(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown train option: " + arg);
        }
    }

    if (options.data_path.empty()) {
        throw std::invalid_argument("train requires --data <jsonl>");
    }
    if (options.checkpoint_path.empty()) {
        throw std::invalid_argument("train requires --checkpoint <path>");
    }
    if (options.vocab_size == 0) {
        throw std::invalid_argument("train requires --vocab-size greater than zero");
    }
    return options;
}

PrepareTextCommandOptions ParsePrepareTextOptions(int argc, char** argv)
{
    PrepareTextCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            options.config_path = ReadOptionValue(argc, argv, i, arg);
        }
    }
    if (!options.config_path.empty()) {
        ApplyPrepareTextConfig(options.config_path, &options);
    }

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--input") {
            options.input_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--output") {
            options.output_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--tokenizer") {
            options.tokenizer = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--vocab-size") {
            options.vocab_size = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--vocab-in") {
            options.vocab_input_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--vocab-out") {
            options.vocab_output_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--seq-len") {
            options.sequence_length = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--max-samples") {
            options.max_samples = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown prepare-text option: " + arg);
        }
    }

    if (options.input_path.empty()) {
        throw std::invalid_argument("prepare-text requires --input <txt>");
    }
    if (options.output_path.empty()) {
        throw std::invalid_argument("prepare-text requires --output <jsonl>");
    }
    if (options.tokenizer != kTokenizerVocab4 && options.tokenizer != kTokenizerByte) {
        throw std::invalid_argument("prepare-text --tokenizer must be vocab4 or byte");
    }
    return options;
}

PipelineCommandOptions ParsePipelineOptions(int argc, char** argv)
{
    PipelineCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            options.config_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--profile") {
            options.profile_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--profile-name") {
            options.profile_name = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown pipeline option: " + arg);
        }
    }
    if (options.config_path.empty() && options.profile_path.empty()) {
        throw std::invalid_argument("pipeline requires --config <json> or --profile <json> --profile-name <name>");
    }
    if (!options.profile_path.empty() || !options.profile_name.empty()) {
        if (options.profile_path.empty() || options.profile_name.empty()) {
            throw std::invalid_argument("pipeline profile requires --profile and --profile-name");
        }
        ApplyPipelineProfile(options.profile_path, options.profile_name, &options);
    }
    if (!options.config_path.empty()) {
        ApplyPipelineConfig(options.config_path, &options);
    }

    if (options.prepare.input_path.empty()) {
        throw std::invalid_argument("pipeline config requires prepare_input");
    }
    if (options.prepare.output_path.empty()) {
        throw std::invalid_argument("pipeline config requires prepare_output");
    }
    if (options.prepare.tokenizer != kTokenizerVocab4 && options.prepare.tokenizer != kTokenizerByte) {
        throw std::invalid_argument("pipeline prepare_tokenizer must be vocab4 or byte");
    }
    if (options.train.data_path.empty()) {
        throw std::invalid_argument("pipeline config requires train_data or prepare_output");
    }
    if (options.train.checkpoint_path.empty()) {
        throw std::invalid_argument("pipeline config requires train_checkpoint");
    }
    if (options.train.vocab_size == 0) {
        throw std::invalid_argument("pipeline train_vocab_size must be greater than zero");
    }
    return options;
}

InspectDatasetCommandOptions ParseInspectDatasetOptions(int argc, char** argv)
{
    InspectDatasetCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--data") {
            options.data_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--vocab-size") {
            options.vocab_size = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown inspect-dataset option: " + arg);
        }
    }
    if (options.data_path.empty()) {
        throw std::invalid_argument("inspect-dataset requires --data <jsonl>");
    }
    if (options.vocab_size == 0) {
        throw std::invalid_argument("inspect-dataset requires --vocab-size greater than zero");
    }
    return options;
}

InspectTextCommandOptions ParseInspectTextOptions(int argc, char** argv)
{
    InspectTextCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input") {
            options.input_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--tokenizer") {
            options.tokenizer = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--vocab-size") {
            options.vocab_size = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--vocab-in") {
            options.vocab_input_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown inspect-text option: " + arg);
        }
    }
    if (options.input_path.empty()) {
        throw std::invalid_argument("inspect-text requires --input <txt>");
    }
    if (options.tokenizer != kTokenizerVocab4 && options.tokenizer != kTokenizerByte) {
        throw std::invalid_argument("inspect-text --tokenizer must be vocab4 or byte");
    }
    if (options.vocab_size == 0) {
        throw std::invalid_argument("inspect-text requires --vocab-size greater than zero");
    }
    return options;
}

PrepareKgCommandOptions ParsePrepareKgOptions(int argc, char** argv)
{
    PrepareKgCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input") {
            options.input_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--output") {
            options.output_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--entity-map") {
            options.entity_map_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--relation-map") {
            options.relation_map_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--max-triples") {
            options.max_triples = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown prepare-kg option: " + arg);
        }
    }
    if (options.input_path.empty()) {
        throw std::invalid_argument("prepare-kg requires --input <tsv>");
    }
    if (options.output_path.empty()) {
        throw std::invalid_argument("prepare-kg requires --output <tsv>");
    }
    return options;
}

PrepareKgPipelineCommandOptions ParsePrepareKgPipelineOptions(int argc, char** argv)
{
    PrepareKgPipelineCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            options.profile_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--profile-name") {
            options.profile_name = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--target") {
            options.target = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--model") {
            options.model_name = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown prepare-kg-pipeline option: " + arg);
        }
    }
    if (options.profile_path.empty() || options.profile_name.empty()) {
        throw std::invalid_argument("prepare-kg-pipeline requires --profile and --profile-name");
    }
    if (options.target != "kg" && options.target != "rgcn") {
        throw std::invalid_argument("prepare-kg-pipeline --target must be kg or rgcn");
    }

    ApplyPrepareKgPipelineProfile(options.profile_path, options.profile_name, &options);
    options.train_kg.model_name = options.model_name;

    if (options.prepare.input_path.empty() || options.prepare.output_path.empty()) {
        throw std::invalid_argument("prepare-kg-pipeline profile requires input and output");
    }
    if (options.target == "kg" && options.train_kg.checkpoint_path.empty()) {
        throw std::invalid_argument("prepare-kg-pipeline KG target requires checkpoint");
    }
    if (options.target == "rgcn" && options.train_rgcn.checkpoint_path.empty()) {
        throw std::invalid_argument("prepare-kg-pipeline R-GCN target requires checkpoint");
    }
    if (options.train_kg.entity_count == 0 || options.train_kg.relation_count == 0 || options.train_kg.embedding_dim == 0) {
        throw std::invalid_argument("prepare-kg-pipeline profile requires entities, relations, and dim");
    }
    return options;
}

InspectKgDatasetCommandOptions ParseInspectKgDatasetOptions(int argc, char** argv)
{
    InspectKgDatasetCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--data") {
            options.data_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--entities") {
            options.entity_count = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--relations") {
            options.relation_count = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown inspect-kg-dataset option: " + arg);
        }
    }
    if (options.data_path.empty()) {
        throw std::invalid_argument("inspect-kg-dataset requires --data <tsv>");
    }
    return options;
}

InspectCheckpointCommandOptions ParseInspectCheckpointOptions(int argc, char** argv)
{
    InspectCheckpointCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--checkpoint") {
            options.checkpoint_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--vocab-size") {
            options.vocab_size = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--hidden-size") {
            options.hidden_size = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown inspect-checkpoint option: " + arg);
        }
    }
    if (options.checkpoint_path.empty()) {
        throw std::invalid_argument("inspect-checkpoint requires --checkpoint <path>");
    }
    if (options.vocab_size == 0) {
        throw std::invalid_argument("inspect-checkpoint requires --vocab-size greater than zero");
    }
    if (options.hidden_size == 0) {
        throw std::invalid_argument("inspect-checkpoint requires --hidden-size greater than zero");
    }
    return options;
}

GenerateCheckpointCommandOptions ParseGenerateCheckpointOptions(int argc, char** argv)
{
    GenerateCheckpointCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--checkpoint") {
            options.checkpoint_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--input") {
            options.input = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--vocab-size") {
            options.vocab_size = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--hidden-size") {
            options.hidden_size = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--max-new") {
            options.max_new_tokens = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--eos") {
            options.eos_token_id = static_cast<int>(ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg));
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown generate-checkpoint option: " + arg);
        }
    }
    if (options.checkpoint_path.empty()) {
        throw std::invalid_argument("generate-checkpoint requires --checkpoint <path>");
    }
    if (options.vocab_size == 0 || options.hidden_size == 0) {
        throw std::invalid_argument("generate-checkpoint dimensions must be greater than zero");
    }
    if (options.max_new_tokens == 0) {
        throw std::invalid_argument("generate-checkpoint requires --max-new greater than zero");
    }
    if (options.eos_token_id < 0) {
        options.eos_token_id = static_cast<int>(options.vocab_size);
    }
    return options;
}

ValidateLmProfileCommandOptions ParseValidateLmProfileOptions(int argc, char** argv)
{
    ValidateLmProfileCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            options.profile_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--profile-name") {
            options.profile_name = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--input") {
            options.input = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--max-new") {
            options.max_new_tokens = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown validate-lm-profile option: " + arg);
        }
    }
    if (options.profile_path.empty() || options.profile_name.empty()) {
        throw std::invalid_argument("validate-lm-profile requires --profile and --profile-name");
    }
    if (options.max_new_tokens == 0) {
        throw std::invalid_argument("validate-lm-profile requires --max-new greater than zero");
    }
    return options;
}

ValidateKgProfileCommandOptions ParseValidateKgProfileOptions(int argc, char** argv)
{
    ValidateKgProfileCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            options.profile_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--profile-name") {
            options.profile_name = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--model") {
            options.model_name = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown validate-kg-profile option: " + arg);
        }
    }
    if (options.profile_path.empty() || options.profile_name.empty()) {
        throw std::invalid_argument("validate-kg-profile requires --profile and --profile-name");
    }
    return options;
}

ValidateRgcnProfileCommandOptions ParseValidateRgcnProfileOptions(int argc, char** argv)
{
    ValidateRgcnProfileCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            options.profile_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--profile-name") {
            options.profile_name = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown validate-rgcn-profile option: " + arg);
        }
    }
    if (options.profile_path.empty() || options.profile_name.empty()) {
        throw std::invalid_argument("validate-rgcn-profile requires --profile and --profile-name");
    }
    return options;
}

TrainKgCommandOptions ParseTrainKgOptions(int argc, char** argv)
{
    TrainKgCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            options.profile_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--profile-name") {
            options.profile_name = ReadOptionValue(argc, argv, i, arg);
        }
    }
    if (!options.profile_path.empty() || !options.profile_name.empty()) {
        if (options.profile_path.empty() || options.profile_name.empty()) {
            throw std::invalid_argument("train-kg profile requires --profile and --profile-name");
        }
        ApplyTrainKgProfile(options.profile_path, options.profile_name, &options);
    }

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile" || arg == "--profile-name") {
            ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--data") {
            options.data_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--checkpoint") {
            options.checkpoint_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--model") {
            options.model_name = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--entities") {
            options.entity_count = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--relations") {
            options.relation_count = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--dim") {
            options.embedding_dim = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--epochs") {
            options.epochs = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--max-triples") {
            options.max_triples = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--learning-rate") {
            options.learning_rate = ParseFloat(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--margin") {
            options.margin = ParseFloat(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown train-kg option: " + arg);
        }
    }
    if (options.data_path.empty()) {
        throw std::invalid_argument("train-kg requires --data <tsv>");
    }
    if (options.entity_count == 0) {
        throw std::invalid_argument("train-kg requires --entities greater than zero");
    }
    if (options.relation_count == 0) {
        throw std::invalid_argument("train-kg requires --relations greater than zero");
    }
    if (options.embedding_dim == 0) {
        throw std::invalid_argument("train-kg requires --dim greater than zero");
    }
    return options;
}

InspectKgCheckpointCommandOptions ParseInspectKgCheckpointOptions(int argc, char** argv)
{
    InspectKgCheckpointCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--checkpoint") {
            options.checkpoint_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--model") {
            options.model_name = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--entities") {
            options.entity_count = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--relations") {
            options.relation_count = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--dim") {
            options.embedding_dim = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown inspect-kg-checkpoint option: " + arg);
        }
    }
    if (options.checkpoint_path.empty()) {
        throw std::invalid_argument("inspect-kg-checkpoint requires --checkpoint <path>");
    }
    if (options.entity_count == 0) {
        throw std::invalid_argument("inspect-kg-checkpoint requires --entities greater than zero");
    }
    if (options.relation_count == 0) {
        throw std::invalid_argument("inspect-kg-checkpoint requires --relations greater than zero");
    }
    if (options.embedding_dim == 0) {
        throw std::invalid_argument("inspect-kg-checkpoint requires --dim greater than zero");
    }
    return options;
}

KgPipelineCommandOptions ParseKgPipelineOptions(int argc, char** argv)
{
    KgPipelineCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            options.train.profile_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--profile-name") {
            options.train.profile_name = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--model") {
            options.train.model_name = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown kg-pipeline option: " + arg);
        }
    }
    if (options.train.profile_path.empty() || options.train.profile_name.empty()) {
        throw std::invalid_argument("kg-pipeline requires --profile and --profile-name");
    }
    ApplyTrainKgProfile(options.train.profile_path, options.train.profile_name, &options.train);
    if (options.train.checkpoint_path.empty()) {
        throw std::invalid_argument("kg-pipeline profile requires checkpoint");
    }
    if (options.train.data_path.empty()) {
        throw std::invalid_argument("kg-pipeline profile requires data");
    }
    if (options.train.entity_count == 0 || options.train.relation_count == 0 || options.train.embedding_dim == 0) {
        throw std::invalid_argument("kg-pipeline profile requires entities, relations, and dim");
    }
    return options;
}

TrainRgcnCommandOptions ParseTrainRgcnOptions(int argc, char** argv)
{
    TrainRgcnCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            options.profile_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--profile-name") {
            options.profile_name = ReadOptionValue(argc, argv, i, arg);
        }
    }
    if (!options.profile_path.empty() || !options.profile_name.empty()) {
        if (options.profile_path.empty() || options.profile_name.empty()) {
            throw std::invalid_argument("train-rgcn profile requires --profile and --profile-name");
        }
        ApplyTrainRgcnProfile(options.profile_path, options.profile_name, &options);
    }

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile" || arg == "--profile-name") {
            ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--data") {
            options.data_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--checkpoint") {
            options.checkpoint_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--nodes") {
            options.node_count = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--relations") {
            options.relation_count = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--dim") {
            options.feature_dim = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--epochs") {
            options.epochs = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--max-triples") {
            options.max_triples = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--learning-rate") {
            options.learning_rate = ParseFloat(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown train-rgcn option: " + arg);
        }
    }

    if (options.data_path.empty()) {
        throw std::invalid_argument("train-rgcn requires --data <tsv>");
    }
    if (options.node_count == 0) {
        throw std::invalid_argument("train-rgcn requires --nodes greater than zero");
    }
    if (options.relation_count == 0) {
        throw std::invalid_argument("train-rgcn requires --relations greater than zero");
    }
    if (options.feature_dim == 0) {
        throw std::invalid_argument("train-rgcn requires --dim greater than zero");
    }
    return options;
}

InspectRgcnCheckpointCommandOptions ParseInspectRgcnCheckpointOptions(int argc, char** argv)
{
    InspectRgcnCheckpointCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--checkpoint") {
            options.checkpoint_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--nodes") {
            options.node_count = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--relations") {
            options.relation_count = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--dim") {
            options.feature_dim = ParseSizeT(ReadOptionValue(argc, argv, i, arg), arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown inspect-rgcn-checkpoint option: " + arg);
        }
    }
    if (options.checkpoint_path.empty()) {
        throw std::invalid_argument("inspect-rgcn-checkpoint requires --checkpoint <path>");
    }
    if (options.node_count == 0) {
        throw std::invalid_argument("inspect-rgcn-checkpoint requires --nodes greater than zero");
    }
    if (options.relation_count == 0) {
        throw std::invalid_argument("inspect-rgcn-checkpoint requires --relations greater than zero");
    }
    if (options.feature_dim == 0) {
        throw std::invalid_argument("inspect-rgcn-checkpoint requires --dim greater than zero");
    }
    return options;
}

RgcnPipelineCommandOptions ParseRgcnPipelineOptions(int argc, char** argv)
{
    RgcnPipelineCommandOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            options.train.profile_path = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--profile-name") {
            options.train.profile_name = ReadOptionValue(argc, argv, i, arg);
        } else if (arg == "--json") {
            options.json_output = true;
        } else {
            throw std::invalid_argument("unknown rgcn-pipeline option: " + arg);
        }
    }
    if (options.train.profile_path.empty() || options.train.profile_name.empty()) {
        throw std::invalid_argument("rgcn-pipeline requires --profile and --profile-name");
    }
    ApplyTrainRgcnProfile(options.train.profile_path, options.train.profile_name, &options.train);
    if (options.train.checkpoint_path.empty()) {
        throw std::invalid_argument("rgcn-pipeline profile requires checkpoint");
    }
    if (options.train.data_path.empty()) {
        throw std::invalid_argument("rgcn-pipeline profile requires data");
    }
    if (options.train.node_count == 0 || options.train.relation_count == 0 || options.train.feature_dim == 0) {
        throw std::invalid_argument("rgcn-pipeline profile requires nodes, relations, and dim");
    }
    return options;
}

std::string ReadTextFile(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open text input: " + path);
    }
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

int RunGenerateCommand(const std::string& input)
{
    minmind::TokenIds ids = minmind::ParseTokenIds(input);

    minmind::MiniMindConfig config;
    minmind::MiniMindModel model(config);
    const std::vector<float> logits = model.ForwardLastToken(ids);
    const int next = minmind::GreedySample(logits);

    ids.push_back(next);
    std::cout << minmind::JoinTokenIds(ids) << '\n';
    return 0;
}

int RunPrepareTextCommand(const PrepareTextCommandOptions& options)
{
    const std::string text = ReadTextFile(options.input_path);
    minmind::TokenIds token_ids;
    std::size_t vocab_size = kTinyVocabSize;
    if (options.tokenizer == kTokenizerByte) {
        const minmind::ByteVocab vocab = options.vocab_input_path.empty()
            ? minmind::BuildByteVocab(text, options.vocab_size)
            : minmind::LoadByteVocab(options.vocab_input_path);
        if (!options.vocab_output_path.empty()) {
            EnsureParentDirectory(options.vocab_output_path);
            minmind::SaveByteVocab(options.vocab_output_path, vocab);
        }
        token_ids = minmind::EncodeBytesWithVocab(text, vocab);
        vocab_size = vocab.id_to_byte.size() + 1;
    } else {
        token_ids = minmind::EncodeVocab4Chars(text);
    }
    const std::vector<minmind::TrainSample> samples =
        minmind::BuildNextTokenTrainSamples(token_ids, options.sequence_length, options.max_samples);

    EnsureParentDirectory(options.output_path);
    minmind::SaveJsonlTrainSamples(options.output_path, samples);

    if (options.json_output) {
        std::cout << std::setprecision(9);
        std::cout << "{\"ok\":true,\"command\":\"prepare-text\",";
        WriteJsonStringField(std::cout, "input", options.input_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "output", options.output_path);
        std::cout << ",\"tokens\":" << token_ids.size()
                  << ",\"samples\":" << samples.size()
                  << ",\"seq_len\":" << options.sequence_length << ',';
        WriteJsonStringField(std::cout, "tokenizer", options.tokenizer);
        std::cout << ",\"vocab_size\":" << vocab_size;
        if (!options.vocab_output_path.empty()) {
            std::cout << ',';
            WriteJsonStringField(std::cout, "vocab_out", options.vocab_output_path);
        }
        std::cout << "}\n";
        return 0;
    }

    std::cout << "input=" << options.input_path << '\n';
    std::cout << "output=" << options.output_path << '\n';
    std::cout << "tokens=" << token_ids.size() << '\n';
    std::cout << "samples=" << samples.size() << '\n';
    std::cout << "seq_len=" << options.sequence_length << '\n';
    std::cout << "tokenizer=" << options.tokenizer << '\n';
    std::cout << "vocab_size=" << vocab_size << '\n';
    return 0;
}

int RunTrainCommand(const TrainCommandOptions& options)
{
    const std::vector<minmind::TrainSample> samples =
        minmind::LoadJsonlTrainSamples(options.data_path);
    minmind::CausalLmWeights weights = MakeTinyWeights(options.vocab_size);

    const minmind::TrainLoopResult result = minmind::TrainLmHeadSamplesNoCache(
        samples,
        weights,
        options.vocab_size,
        kTinyHiddenSize,
        kTinyNumHeads,
        kTinyIntermediateSize,
        kTinyRmsEps,
        kTinyRopeTheta,
        kIgnoreIndex,
        options.learning_rate,
        options.epochs);

    EnsureParentDirectory(options.checkpoint_path);
    minmind::SaveLmHeadCheckpoint(
        options.checkpoint_path,
        weights,
        options.vocab_size,
        kTinyHiddenSize);

    std::cout << std::setprecision(9);
    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"train\","
                  << "\"samples\":" << samples.size()
                  << ",\"vocab_size\":" << options.vocab_size
                  << ",\"epochs\":" << options.epochs
                  << ",\"learning_rate\":" << options.learning_rate
                  << ",\"loss_history\":[";
        for (std::size_t i = 0; i < result.loss_history.size(); ++i) {
            if (i != 0) {
                std::cout << ',';
            }
            std::cout << result.loss_history[i];
        }
        std::cout << "],";
        WriteJsonStringField(std::cout, "checkpoint", options.checkpoint_path);
        if (!options.profile_name.empty()) {
            std::cout << ',';
            WriteJsonStringField(std::cout, "profile", options.profile_path);
            std::cout << ',';
            WriteJsonStringField(std::cout, "profile_name", options.profile_name);
        }
        std::cout << "}\n";
        return 0;
    }

    std::cout << "samples=" << samples.size() << '\n';
    std::cout << "vocab_size=" << options.vocab_size << '\n';
    std::cout << "epochs=" << options.epochs << '\n';
    for (std::size_t i = 0; i < result.loss_history.size(); ++i) {
        std::cout << "loss[" << i << "]=" << result.loss_history[i] << '\n';
    }
    std::cout << "checkpoint=" << options.checkpoint_path << '\n';
    return 0;
}

int RunInspectDatasetCommand(const InspectDatasetCommandOptions& options)
{
    const std::vector<minmind::TrainSample> samples =
        minmind::LoadJsonlTrainSamples(options.data_path);

    std::size_t min_seq_len = std::numeric_limits<std::size_t>::max();
    std::size_t max_seq_len = 0;
    int max_token_id = 0;
    int max_label_id = 0;
    std::size_t ignored_labels = 0;
    std::size_t supervised_labels = 0;

    for (const minmind::TrainSample& sample : samples) {
        min_seq_len = std::min(min_seq_len, sample.input_ids.size());
        max_seq_len = std::max(max_seq_len, sample.input_ids.size());

        for (const int token : sample.input_ids) {
            if (token < 0 || static_cast<std::size_t>(token) >= options.vocab_size) {
                throw std::runtime_error("dataset input token id is outside vocabulary");
            }
            max_token_id = std::max(max_token_id, token);
        }
        for (const int label : sample.labels) {
            if (label == kIgnoreIndex) {
                ++ignored_labels;
                continue;
            }
            if (label < 0 || static_cast<std::size_t>(label) >= options.vocab_size) {
                throw std::runtime_error("dataset label id is outside vocabulary");
            }
            ++supervised_labels;
            max_label_id = std::max(max_label_id, label);
        }
    }

    if (supervised_labels == 0) {
        throw std::runtime_error("dataset has no supervised labels");
    }

    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"inspect-dataset\",";
        WriteJsonStringField(std::cout, "data", options.data_path);
        std::cout << ",\"samples\":" << samples.size()
                  << ",\"min_seq_len\":" << min_seq_len
                  << ",\"max_seq_len\":" << max_seq_len
                  << ",\"vocab_size\":" << options.vocab_size
                  << ",\"max_token_id\":" << max_token_id
                  << ",\"max_label_id\":" << max_label_id
                  << ",\"ignored_labels\":" << ignored_labels
                  << ",\"supervised_labels\":" << supervised_labels
                  << "}\n";
        return 0;
    }

    std::cout << "data=" << options.data_path << '\n';
    std::cout << "samples=" << samples.size() << '\n';
    std::cout << "min_seq_len=" << min_seq_len << '\n';
    std::cout << "max_seq_len=" << max_seq_len << '\n';
    std::cout << "vocab_size=" << options.vocab_size << '\n';
    std::cout << "max_token_id=" << max_token_id << '\n';
    std::cout << "max_label_id=" << max_label_id << '\n';
    std::cout << "ignored_labels=" << ignored_labels << '\n';
    std::cout << "supervised_labels=" << supervised_labels << '\n';
    return 0;
}

int RunInspectTextCommand(const InspectTextCommandOptions& options)
{
    const std::string text = ReadTextFile(options.input_path);
    minmind::TokenIds token_ids;
    std::size_t effective_vocab_size = kTinyVocabSize;
    if (options.tokenizer == kTokenizerByte) {
        const minmind::ByteVocab vocab = options.vocab_input_path.empty()
            ? minmind::BuildByteVocab(text, options.vocab_size)
            : minmind::LoadByteVocab(options.vocab_input_path);
        token_ids = minmind::EncodeBytesWithVocab(text, vocab);
        effective_vocab_size = vocab.id_to_byte.size() + 1;
    } else {
        token_ids = minmind::EncodeVocab4Chars(text);
        effective_vocab_size = kTinyVocabSize;
    }

    int max_token_id = 0;
    for (const int token : token_ids) {
        max_token_id = std::max(max_token_id, token);
    }

    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"inspect-text\",";
        WriteJsonStringField(std::cout, "input", options.input_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "tokenizer", options.tokenizer);
        std::cout << ",\"bytes\":" << text.size()
                  << ",\"tokens\":" << token_ids.size()
                  << ",\"vocab_size\":" << effective_vocab_size
                  << ",\"max_token_id\":" << max_token_id;
        if (!options.vocab_input_path.empty()) {
            std::cout << ',';
            WriteJsonStringField(std::cout, "vocab_in", options.vocab_input_path);
        }
        std::cout << "}\n";
        return 0;
    }

    std::cout << "input=" << options.input_path << '\n';
    std::cout << "tokenizer=" << options.tokenizer << '\n';
    std::cout << "bytes=" << text.size() << '\n';
    std::cout << "tokens=" << token_ids.size() << '\n';
    std::cout << "vocab_size=" << effective_vocab_size << '\n';
    std::cout << "max_token_id=" << max_token_id << '\n';
    return 0;
}

std::size_t GetOrAddId(
    const std::string& value,
    std::unordered_map<std::string, std::size_t>* ids,
    std::vector<std::string>* names)
{
    const auto found = ids->find(value);
    if (found != ids->end()) {
        return found->second;
    }
    const std::size_t id = names->size();
    ids->emplace(value, id);
    names->push_back(value);
    return id;
}

void SaveIdMap(const std::string& path, const std::vector<std::string>& names)
{
    EnsureParentDirectory(path);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open id map for writing: " + path);
    }
    for (std::size_t id = 0; id < names.size(); ++id) {
        output << id << '\t' << names[id] << '\n';
    }
    if (!output) {
        throw std::runtime_error("failed to write id map: " + path);
    }
}

int RunPrepareKgCommand(const PrepareKgCommandOptions& options)
{
    std::ifstream input(options.input_path);
    if (!input) {
        throw std::runtime_error("failed to open KG input: " + options.input_path);
    }
    EnsureParentDirectory(options.output_path);
    std::ofstream output(options.output_path);
    if (!output) {
        throw std::runtime_error("failed to open KG output: " + options.output_path);
    }

    std::unordered_map<std::string, std::size_t> entity_ids;
    std::unordered_map<std::string, std::size_t> relation_ids;
    std::vector<std::string> entity_names;
    std::vector<std::string> relation_names;
    std::size_t triples = 0;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (options.max_triples > 0 && triples >= options.max_triples) {
            break;
        }

        std::istringstream parsed(line);
        std::string head;
        std::string relation;
        std::string tail;
        parsed >> head >> relation >> tail;
        if (!parsed) {
            throw std::runtime_error("invalid KG input triple line: " + line);
        }

        const std::size_t head_id = GetOrAddId(head, &entity_ids, &entity_names);
        const std::size_t tail_id = GetOrAddId(tail, &entity_ids, &entity_names);
        const std::size_t relation_id = GetOrAddId(relation, &relation_ids, &relation_names);
        output << head_id << '\t' << relation_id << '\t' << tail_id << '\n';
        ++triples;
    }
    if (triples == 0) {
        throw std::runtime_error("prepare-kg produced no triples");
    }
    if (!output) {
        throw std::runtime_error("failed to write KG output: " + options.output_path);
    }

    if (!options.entity_map_path.empty()) {
        SaveIdMap(options.entity_map_path, entity_names);
    }
    if (!options.relation_map_path.empty()) {
        SaveIdMap(options.relation_map_path, relation_names);
    }

    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"prepare-kg\",";
        WriteJsonStringField(std::cout, "input", options.input_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "output", options.output_path);
        std::cout << ",\"triples\":" << triples
                  << ",\"entities\":" << entity_names.size()
                  << ",\"relations\":" << relation_names.size()
                  << ",\"max_triples\":" << options.max_triples;
        if (!options.entity_map_path.empty()) {
            std::cout << ',';
            WriteJsonStringField(std::cout, "entity_map", options.entity_map_path);
        }
        if (!options.relation_map_path.empty()) {
            std::cout << ',';
            WriteJsonStringField(std::cout, "relation_map", options.relation_map_path);
        }
        std::cout << "}\n";
        return 0;
    }

    std::cout << "input=" << options.input_path << '\n';
    std::cout << "output=" << options.output_path << '\n';
    std::cout << "triples=" << triples << '\n';
    std::cout << "entities=" << entity_names.size() << '\n';
    std::cout << "relations=" << relation_names.size() << '\n';
    return 0;
}

int RunInspectKgDatasetCommand(const InspectKgDatasetCommandOptions& options)
{
    const std::vector<minmind::KgTriple> triples = minmind::LoadKgTriplesTsv(options.data_path);
    std::size_t max_entity_id = 0;
    std::size_t max_relation_id = 0;
    for (const minmind::KgTriple& triple : triples) {
        max_entity_id = std::max(max_entity_id, std::max(triple.head, triple.tail));
        max_relation_id = std::max(max_relation_id, triple.relation);
        if (options.entity_count > 0 &&
            (triple.head >= options.entity_count || triple.tail >= options.entity_count)) {
            throw std::runtime_error("KG dataset entity id is outside configured entity count");
        }
        if (options.relation_count > 0 && triple.relation >= options.relation_count) {
            throw std::runtime_error("KG dataset relation id is outside configured relation count");
        }
    }

    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"inspect-kg-dataset\",";
        WriteJsonStringField(std::cout, "data", options.data_path);
        std::cout << ",\"triples\":" << triples.size()
                  << ",\"max_entity_id\":" << max_entity_id
                  << ",\"max_relation_id\":" << max_relation_id
                  << ",\"entities\":" << options.entity_count
                  << ",\"relations\":" << options.relation_count
                  << "}\n";
        return 0;
    }

    std::cout << "data=" << options.data_path << '\n';
    std::cout << "triples=" << triples.size() << '\n';
    std::cout << "max_entity_id=" << max_entity_id << '\n';
    std::cout << "max_relation_id=" << max_relation_id << '\n';
    std::cout << "entities=" << options.entity_count << '\n';
    std::cout << "relations=" << options.relation_count << '\n';
    return 0;
}

int RunInspectCheckpointCommand(const InspectCheckpointCommandOptions& options)
{
    minmind::CausalLmWeights weights;
    weights.lm_head_weight.resize(options.vocab_size * options.hidden_size, 0.0f);
    minmind::LoadLmHeadCheckpoint(
        options.checkpoint_path,
        weights,
        options.vocab_size,
        options.hidden_size);

    float min_weight = weights.lm_head_weight.empty() ? 0.0f : weights.lm_head_weight.front();
    float max_weight = min_weight;
    double sum = 0.0;
    for (const float value : weights.lm_head_weight) {
        min_weight = std::min(min_weight, value);
        max_weight = std::max(max_weight, value);
        sum += value;
    }
    const double mean = weights.lm_head_weight.empty()
        ? 0.0
        : sum / static_cast<double>(weights.lm_head_weight.size());

    std::cout << std::setprecision(9);
    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"inspect-checkpoint\",";
        WriteJsonStringField(std::cout, "checkpoint", options.checkpoint_path);
        std::cout << ",\"vocab_size\":" << options.vocab_size
                  << ",\"hidden_size\":" << options.hidden_size
                  << ",\"weight_count\":" << weights.lm_head_weight.size()
                  << ",\"min_weight\":" << min_weight
                  << ",\"max_weight\":" << max_weight
                  << ",\"mean_weight\":" << mean
                  << "}\n";
        return 0;
    }

    std::cout << "checkpoint=" << options.checkpoint_path << '\n';
    std::cout << "vocab_size=" << options.vocab_size << '\n';
    std::cout << "hidden_size=" << options.hidden_size << '\n';
    std::cout << "weight_count=" << weights.lm_head_weight.size() << '\n';
    std::cout << "min_weight=" << min_weight << '\n';
    std::cout << "max_weight=" << max_weight << '\n';
    std::cout << "mean_weight=" << mean << '\n';
    return 0;
}

int RunGenerateCheckpointCommand(const GenerateCheckpointCommandOptions& options)
{
    minmind::TokenIds input_ids = minmind::ParseTokenIds(options.input);
    if (input_ids.empty()) {
        throw std::invalid_argument("generate-checkpoint requires non-empty input token ids");
    }

    minmind::CausalLmWeights weights = MakeTinyWeights(options.vocab_size);
    minmind::LoadLmHeadCheckpoint(
        options.checkpoint_path,
        weights,
        options.vocab_size,
        options.hidden_size);

    const minmind::TokenIds generated = minmind::CausalLmGenerateGreedyNoCache(
        input_ids,
        weights,
        options.vocab_size,
        options.hidden_size,
        kTinyNumHeads,
        kTinyIntermediateSize,
        kTinyRmsEps,
        kTinyRopeTheta,
        options.max_new_tokens,
        options.eos_token_id);

    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"generate-checkpoint\",";
        WriteJsonStringField(std::cout, "checkpoint", options.checkpoint_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "input", minmind::JoinTokenIds(input_ids));
        std::cout << ',';
        WriteJsonStringField(std::cout, "generated", minmind::JoinTokenIds(generated));
        std::cout << ",\"vocab_size\":" << options.vocab_size
                  << ",\"hidden_size\":" << options.hidden_size
                  << ",\"max_new\":" << options.max_new_tokens
                  << ",\"generated_tokens\":" << (generated.size() - input_ids.size())
                  << "}\n";
        return 0;
    }

    std::cout << minmind::JoinTokenIds(generated) << '\n';
    return 0;
}

int RunValidateLmProfileCommand(const ValidateLmProfileCommandOptions& options)
{
    TrainCommandOptions train;
    ApplyTrainProfile(options.profile_path, options.profile_name, &train);
    if (train.data_path.empty() || train.checkpoint_path.empty()) {
        throw std::invalid_argument("validate-lm-profile requires profile data and checkpoint");
    }

    InspectDatasetCommandOptions inspect_data;
    inspect_data.data_path = train.data_path;
    inspect_data.vocab_size = train.vocab_size;

    InspectCheckpointCommandOptions inspect_checkpoint;
    inspect_checkpoint.checkpoint_path = train.checkpoint_path;
    inspect_checkpoint.vocab_size = train.vocab_size;
    inspect_checkpoint.hidden_size = kTinyHiddenSize;

    GenerateCheckpointCommandOptions generate;
    generate.checkpoint_path = train.checkpoint_path;
    generate.input = options.input;
    generate.vocab_size = train.vocab_size;
    generate.hidden_size = kTinyHiddenSize;
    generate.max_new_tokens = options.max_new_tokens;

    std::ostringstream sink;
    std::streambuf* old_buffer = std::cout.rdbuf(sink.rdbuf());
    try {
        RunInspectDatasetCommand(inspect_data);
        RunInspectCheckpointCommand(inspect_checkpoint);
        RunGenerateCheckpointCommand(generate);
    } catch (...) {
        std::cout.rdbuf(old_buffer);
        throw;
    }
    std::cout.rdbuf(old_buffer);

    if (options.json_output) {
        minmind::TokenIds input_ids = minmind::ParseTokenIds(options.input);
        minmind::CausalLmWeights weights = MakeTinyWeights(train.vocab_size);
        minmind::LoadLmHeadCheckpoint(train.checkpoint_path, weights, train.vocab_size, kTinyHiddenSize);
        const minmind::TokenIds generated = minmind::CausalLmGenerateGreedyNoCache(
            input_ids,
            weights,
            train.vocab_size,
            kTinyHiddenSize,
            kTinyNumHeads,
            kTinyIntermediateSize,
            kTinyRmsEps,
            kTinyRopeTheta,
            options.max_new_tokens,
            static_cast<int>(train.vocab_size));

        std::cout << "{\"ok\":true,\"command\":\"validate-lm-profile\",";
        WriteJsonStringField(std::cout, "profile", options.profile_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "profile_name", options.profile_name);
        std::cout << ',';
        WriteJsonStringField(std::cout, "data", train.data_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "checkpoint", train.checkpoint_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "generated", minmind::JoinTokenIds(generated));
        std::cout << ",\"vocab_size\":" << train.vocab_size
                  << ",\"dataset_inspected\":true"
                  << ",\"checkpoint_inspected\":true"
                  << ",\"generation_validated\":true"
                  << "}\n";
        return 0;
    }

    std::cout << "profile=" << options.profile_path << '\n';
    std::cout << "profile_name=" << options.profile_name << '\n';
    std::cout << "dataset_inspected=true\n";
    std::cout << "checkpoint_inspected=true\n";
    std::cout << "generation_validated=true\n";
    return 0;
}

int RunValidateKgProfileCommand(const ValidateKgProfileCommandOptions& options)
{
    TrainKgCommandOptions train;
    train.profile_path = options.profile_path;
    train.profile_name = options.profile_name;
    train.model_name = options.model_name;
    ApplyTrainKgProfile(options.profile_path, options.profile_name, &train);
    if (train.data_path.empty() || train.checkpoint_path.empty()) {
        throw std::invalid_argument("validate-kg-profile requires profile data and checkpoint");
    }

    InspectKgDatasetCommandOptions inspect_data;
    inspect_data.data_path = train.data_path;
    inspect_data.entity_count = train.entity_count;
    inspect_data.relation_count = train.relation_count;

    InspectKgCheckpointCommandOptions inspect_checkpoint;
    inspect_checkpoint.checkpoint_path = train.checkpoint_path;
    inspect_checkpoint.model_name = train.model_name;
    inspect_checkpoint.entity_count = train.entity_count;
    inspect_checkpoint.relation_count = train.relation_count;
    inspect_checkpoint.embedding_dim = train.embedding_dim;

    std::ostringstream sink;
    std::streambuf* old_buffer = std::cout.rdbuf(sink.rdbuf());
    try {
        RunInspectKgDatasetCommand(inspect_data);
        RunInspectKgCheckpointCommand(inspect_checkpoint);
    } catch (...) {
        std::cout.rdbuf(old_buffer);
        throw;
    }
    std::cout.rdbuf(old_buffer);

    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"validate-kg-profile\",";
        WriteJsonStringField(std::cout, "profile", options.profile_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "profile_name", options.profile_name);
        std::cout << ',';
        WriteJsonStringField(std::cout, "data", train.data_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "model", train.model_name);
        std::cout << ',';
        WriteJsonStringField(std::cout, "checkpoint", train.checkpoint_path);
        std::cout << ",\"entities\":" << train.entity_count
                  << ",\"relations\":" << train.relation_count
                  << ",\"dim\":" << train.embedding_dim
                  << ",\"dataset_inspected\":true"
                  << ",\"checkpoint_inspected\":true"
                  << "}\n";
        return 0;
    }

    std::cout << "profile=" << options.profile_path << '\n';
    std::cout << "profile_name=" << options.profile_name << '\n';
    std::cout << "dataset_inspected=true\n";
    std::cout << "checkpoint_inspected=true\n";
    return 0;
}

int RunValidateRgcnProfileCommand(const ValidateRgcnProfileCommandOptions& options)
{
    TrainRgcnCommandOptions train;
    train.profile_path = options.profile_path;
    train.profile_name = options.profile_name;
    ApplyTrainRgcnProfile(options.profile_path, options.profile_name, &train);
    if (train.data_path.empty() || train.checkpoint_path.empty()) {
        throw std::invalid_argument("validate-rgcn-profile requires profile data and checkpoint");
    }

    InspectKgDatasetCommandOptions inspect_data;
    inspect_data.data_path = train.data_path;
    inspect_data.entity_count = train.node_count;
    inspect_data.relation_count = train.relation_count;

    InspectRgcnCheckpointCommandOptions inspect_checkpoint;
    inspect_checkpoint.checkpoint_path = train.checkpoint_path;
    inspect_checkpoint.node_count = train.node_count;
    inspect_checkpoint.relation_count = train.relation_count;
    inspect_checkpoint.feature_dim = train.feature_dim;

    std::ostringstream sink;
    std::streambuf* old_buffer = std::cout.rdbuf(sink.rdbuf());
    try {
        RunInspectKgDatasetCommand(inspect_data);
        RunInspectRgcnCheckpointCommand(inspect_checkpoint);
    } catch (...) {
        std::cout.rdbuf(old_buffer);
        throw;
    }
    std::cout.rdbuf(old_buffer);

    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"validate-rgcn-profile\",";
        WriteJsonStringField(std::cout, "profile", options.profile_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "profile_name", options.profile_name);
        std::cout << ',';
        WriteJsonStringField(std::cout, "data", train.data_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "checkpoint", train.checkpoint_path);
        std::cout << ",\"nodes\":" << train.node_count
                  << ",\"relations\":" << train.relation_count
                  << ",\"dim\":" << train.feature_dim
                  << ",\"dataset_inspected\":true"
                  << ",\"checkpoint_inspected\":true"
                  << "}\n";
        return 0;
    }

    std::cout << "profile=" << options.profile_path << '\n';
    std::cout << "profile_name=" << options.profile_name << '\n';
    std::cout << "dataset_inspected=true\n";
    std::cout << "checkpoint_inspected=true\n";
    return 0;
}

int RunTrainKgCommand(const TrainKgCommandOptions& options)
{
    minmind::KgEmbeddingConfig config;
    config.model = minmind::ParseKgEmbeddingModel(options.model_name);
    config.entity_count = options.entity_count;
    config.relation_count = options.relation_count;
    config.embedding_dim = options.embedding_dim;
    config.epochs = options.epochs;
    config.learning_rate = options.learning_rate;
    config.margin = options.margin;

    std::vector<minmind::KgTriple> triples = minmind::LoadKgTriplesTsv(options.data_path);
    if (options.max_triples > 0 && triples.size() > options.max_triples) {
        triples.resize(options.max_triples);
    }
    minmind::KgEmbeddingState state = minmind::InitKgEmbeddingState(config);
    const minmind::KgTrainResult result = minmind::TrainKgEmbeddings(triples, config, state);
    if (!options.checkpoint_path.empty()) {
        EnsureParentDirectory(options.checkpoint_path);
        minmind::SaveKgEmbeddingCheckpoint(
            options.checkpoint_path,
            config.model,
            config.entity_count,
            config.relation_count,
            config.embedding_dim,
            state);
    }

    std::cout << std::setprecision(9);
    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"train-kg\",";
        WriteJsonStringField(std::cout, "data", options.data_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "model", minmind::KgEmbeddingModelName(config.model));
        std::cout << ",\"triples\":" << triples.size()
                  << ",\"entities\":" << config.entity_count
                  << ",\"relations\":" << config.relation_count
                  << ",\"dim\":" << config.embedding_dim
                  << ",\"epochs\":" << config.epochs
                  << ",\"max_triples\":" << options.max_triples
                  << ",\"learning_rate\":" << config.learning_rate
                  << ",\"margin\":" << config.margin
                  << ",\"entity_width\":" << state.entity_width
                  << ",\"relation_width\":" << state.relation_width
                  << ",\"loss_history\":[";
        for (std::size_t i = 0; i < result.loss_history.size(); ++i) {
            if (i != 0) {
                std::cout << ',';
            }
            std::cout << result.loss_history[i];
        }
        std::cout << "]";
        if (!options.profile_name.empty()) {
            std::cout << ',';
            WriteJsonStringField(std::cout, "profile", options.profile_path);
            std::cout << ',';
            WriteJsonStringField(std::cout, "profile_name", options.profile_name);
        }
        if (!options.checkpoint_path.empty()) {
            std::cout << ',';
            WriteJsonStringField(std::cout, "checkpoint", options.checkpoint_path);
        }
        std::cout << "}\n";
        return 0;
    }

    std::cout << "data=" << options.data_path << '\n';
    std::cout << "model=" << minmind::KgEmbeddingModelName(config.model) << '\n';
    std::cout << "triples=" << triples.size() << '\n';
    std::cout << "entities=" << config.entity_count << '\n';
    std::cout << "relations=" << config.relation_count << '\n';
    std::cout << "dim=" << config.embedding_dim << '\n';
    std::cout << "max_triples=" << options.max_triples << '\n';
    if (!options.checkpoint_path.empty()) {
        std::cout << "checkpoint=" << options.checkpoint_path << '\n';
    }
    for (std::size_t i = 0; i < result.loss_history.size(); ++i) {
        std::cout << "loss[" << i << "]=" << result.loss_history[i] << '\n';
    }
    return 0;
}

int RunInspectKgCheckpointCommand(const InspectKgCheckpointCommandOptions& options)
{
    const minmind::KgEmbeddingModel model = minmind::ParseKgEmbeddingModel(options.model_name);
    minmind::KgEmbeddingState state;
    minmind::LoadKgEmbeddingCheckpoint(
        options.checkpoint_path,
        model,
        options.entity_count,
        options.relation_count,
        options.embedding_dim,
        state);

    float min_value = 0.0f;
    float max_value = 0.0f;
    double sum = 0.0;
    std::size_t value_count = 0;
    const auto observe = [&](const std::vector<float>& values) {
        for (const float value : values) {
            if (value_count == 0) {
                min_value = value;
                max_value = value;
            } else {
                min_value = std::min(min_value, value);
                max_value = std::max(max_value, value);
            }
            sum += value;
            ++value_count;
        }
    };
    observe(state.entity_embeddings);
    observe(state.relation_embeddings);
    const double mean = value_count == 0 ? 0.0 : sum / static_cast<double>(value_count);

    std::cout << std::setprecision(9);
    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"inspect-kg-checkpoint\",";
        WriteJsonStringField(std::cout, "checkpoint", options.checkpoint_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "model", minmind::KgEmbeddingModelName(model));
        std::cout << ",\"entities\":" << options.entity_count
                  << ",\"relations\":" << options.relation_count
                  << ",\"dim\":" << options.embedding_dim
                  << ",\"entity_width\":" << state.entity_width
                  << ",\"relation_width\":" << state.relation_width
                  << ",\"entity_weight_count\":" << state.entity_embeddings.size()
                  << ",\"relation_weight_count\":" << state.relation_embeddings.size()
                  << ",\"min_weight\":" << min_value
                  << ",\"max_weight\":" << max_value
                  << ",\"mean_weight\":" << mean
                  << "}\n";
        return 0;
    }

    std::cout << "checkpoint=" << options.checkpoint_path << '\n';
    std::cout << "model=" << minmind::KgEmbeddingModelName(model) << '\n';
    std::cout << "entities=" << options.entity_count << '\n';
    std::cout << "relations=" << options.relation_count << '\n';
    std::cout << "dim=" << options.embedding_dim << '\n';
    std::cout << "entity_width=" << state.entity_width << '\n';
    std::cout << "relation_width=" << state.relation_width << '\n';
    std::cout << "entity_weight_count=" << state.entity_embeddings.size() << '\n';
    std::cout << "relation_weight_count=" << state.relation_embeddings.size() << '\n';
    std::cout << "min_weight=" << min_value << '\n';
    std::cout << "max_weight=" << max_value << '\n';
    std::cout << "mean_weight=" << mean << '\n';
    return 0;
}

int RunKgPipelineCommand(const KgPipelineCommandOptions& options)
{
    TrainKgCommandOptions train = options.train;
    train.json_output = false;

    InspectKgCheckpointCommandOptions inspect;
    inspect.checkpoint_path = train.checkpoint_path;
    inspect.model_name = train.model_name;
    inspect.entity_count = train.entity_count;
    inspect.relation_count = train.relation_count;
    inspect.embedding_dim = train.embedding_dim;
    inspect.json_output = false;

    std::ostringstream sink;
    std::streambuf* old_buffer = std::cout.rdbuf(sink.rdbuf());
    try {
        RunTrainKgCommand(train);
        RunInspectKgCheckpointCommand(inspect);
    } catch (...) {
        std::cout.rdbuf(old_buffer);
        throw;
    }
    std::cout.rdbuf(old_buffer);

    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"kg-pipeline\",";
        WriteJsonStringField(std::cout, "profile", train.profile_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "profile_name", train.profile_name);
        std::cout << ',';
        WriteJsonStringField(std::cout, "data", train.data_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "model", train.model_name);
        std::cout << ',';
        WriteJsonStringField(std::cout, "checkpoint", train.checkpoint_path);
        std::cout << ",\"entities\":" << train.entity_count
                  << ",\"relations\":" << train.relation_count
                  << ",\"dim\":" << train.embedding_dim
                  << ",\"epochs\":" << train.epochs
                  << ",\"max_triples\":" << train.max_triples
                  << ",\"checkpoint_inspected\":true"
                  << "}\n";
        return 0;
    }

    std::cout << "profile=" << train.profile_path << '\n';
    std::cout << "profile_name=" << train.profile_name << '\n';
    std::cout << "data=" << train.data_path << '\n';
    std::cout << "model=" << train.model_name << '\n';
    std::cout << "checkpoint=" << train.checkpoint_path << '\n';
    std::cout << "checkpoint_inspected=true\n";
    return 0;
}

int RunTrainRgcnCommand(const TrainRgcnCommandOptions& options)
{
    minmind::RgcnTrainConfig config;
    config.node_count = options.node_count;
    config.relation_count = options.relation_count;
    config.feature_dim = options.feature_dim;
    config.epochs = options.epochs;
    config.learning_rate = options.learning_rate;

    std::vector<minmind::KgTriple> triples = minmind::LoadKgTriplesTsv(options.data_path);
    if (options.max_triples > 0 && triples.size() > options.max_triples) {
        triples.resize(options.max_triples);
    }
    std::vector<float> node_features = minmind::InitRgcnNodeFeatures(config.node_count, config.feature_dim);
    std::vector<float> relation_weights = minmind::InitRgcnRelationWeights(config.relation_count, config.feature_dim);
    const minmind::RgcnTrainResult result = minmind::TrainRgcnRelationMean(
        triples,
        config,
        node_features,
        relation_weights);
    if (!options.checkpoint_path.empty()) {
        EnsureParentDirectory(options.checkpoint_path);
        minmind::SaveRgcnRelationCheckpoint(
            options.checkpoint_path,
            config.node_count,
            config.relation_count,
            config.feature_dim,
            relation_weights);
    }

    std::cout << std::setprecision(9);
    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"train-rgcn\",";
        WriteJsonStringField(std::cout, "data", options.data_path);
        std::cout << ",\"triples\":" << triples.size()
                  << ",\"nodes\":" << config.node_count
                  << ",\"relations\":" << config.relation_count
                  << ",\"dim\":" << config.feature_dim
                  << ",\"epochs\":" << config.epochs
                  << ",\"max_triples\":" << options.max_triples
                  << ",\"learning_rate\":" << config.learning_rate
                  << ",\"relation_weight_count\":" << relation_weights.size()
                  << ",\"loss_history\":[";
        for (std::size_t i = 0; i < result.loss_history.size(); ++i) {
            if (i != 0) {
                std::cout << ',';
            }
            std::cout << result.loss_history[i];
        }
        std::cout << "]";
        if (!options.profile_name.empty()) {
            std::cout << ',';
            WriteJsonStringField(std::cout, "profile", options.profile_path);
            std::cout << ',';
            WriteJsonStringField(std::cout, "profile_name", options.profile_name);
        }
        if (!options.checkpoint_path.empty()) {
            std::cout << ',';
            WriteJsonStringField(std::cout, "checkpoint", options.checkpoint_path);
        }
        std::cout << "}\n";
        return 0;
    }

    std::cout << "data=" << options.data_path << '\n';
    std::cout << "triples=" << triples.size() << '\n';
    std::cout << "nodes=" << config.node_count << '\n';
    std::cout << "relations=" << config.relation_count << '\n';
    std::cout << "dim=" << config.feature_dim << '\n';
    std::cout << "max_triples=" << options.max_triples << '\n';
    if (!options.checkpoint_path.empty()) {
        std::cout << "checkpoint=" << options.checkpoint_path << '\n';
    }
    for (std::size_t i = 0; i < result.loss_history.size(); ++i) {
        std::cout << "loss[" << i << "]=" << result.loss_history[i] << '\n';
    }
    return 0;
}

int RunInspectRgcnCheckpointCommand(const InspectRgcnCheckpointCommandOptions& options)
{
    std::vector<float> relation_weights;
    minmind::LoadRgcnRelationCheckpoint(
        options.checkpoint_path,
        options.node_count,
        options.relation_count,
        options.feature_dim,
        relation_weights);

    float min_value = 0.0f;
    float max_value = 0.0f;
    double sum = 0.0;
    for (std::size_t i = 0; i < relation_weights.size(); ++i) {
        const float value = relation_weights[i];
        if (i == 0) {
            min_value = value;
            max_value = value;
        } else {
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
        sum += value;
    }
    const double mean = relation_weights.empty()
        ? 0.0
        : sum / static_cast<double>(relation_weights.size());

    std::cout << std::setprecision(9);
    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"inspect-rgcn-checkpoint\",";
        WriteJsonStringField(std::cout, "checkpoint", options.checkpoint_path);
        std::cout << ",\"nodes\":" << options.node_count
                  << ",\"relations\":" << options.relation_count
                  << ",\"dim\":" << options.feature_dim
                  << ",\"relation_weight_count\":" << relation_weights.size()
                  << ",\"min_weight\":" << min_value
                  << ",\"max_weight\":" << max_value
                  << ",\"mean_weight\":" << mean
                  << "}\n";
        return 0;
    }

    std::cout << "checkpoint=" << options.checkpoint_path << '\n';
    std::cout << "nodes=" << options.node_count << '\n';
    std::cout << "relations=" << options.relation_count << '\n';
    std::cout << "dim=" << options.feature_dim << '\n';
    std::cout << "relation_weight_count=" << relation_weights.size() << '\n';
    std::cout << "min_weight=" << min_value << '\n';
    std::cout << "max_weight=" << max_value << '\n';
    std::cout << "mean_weight=" << mean << '\n';
    return 0;
}

int RunRgcnPipelineCommand(const RgcnPipelineCommandOptions& options)
{
    TrainRgcnCommandOptions train = options.train;
    train.json_output = false;

    InspectRgcnCheckpointCommandOptions inspect;
    inspect.checkpoint_path = train.checkpoint_path;
    inspect.node_count = train.node_count;
    inspect.relation_count = train.relation_count;
    inspect.feature_dim = train.feature_dim;
    inspect.json_output = false;

    std::ostringstream sink;
    std::streambuf* old_buffer = std::cout.rdbuf(sink.rdbuf());
    try {
        RunTrainRgcnCommand(train);
        RunInspectRgcnCheckpointCommand(inspect);
    } catch (...) {
        std::cout.rdbuf(old_buffer);
        throw;
    }
    std::cout.rdbuf(old_buffer);

    if (options.json_output) {
        std::cout << "{\"ok\":true,\"command\":\"rgcn-pipeline\",";
        WriteJsonStringField(std::cout, "profile", train.profile_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "profile_name", train.profile_name);
        std::cout << ',';
        WriteJsonStringField(std::cout, "data", train.data_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "checkpoint", train.checkpoint_path);
        std::cout << ",\"nodes\":" << train.node_count
                  << ",\"relations\":" << train.relation_count
                  << ",\"dim\":" << train.feature_dim
                  << ",\"epochs\":" << train.epochs
                  << ",\"max_triples\":" << train.max_triples
                  << ",\"checkpoint_inspected\":true"
                  << "}\n";
        return 0;
    }

    std::cout << "profile=" << train.profile_path << '\n';
    std::cout << "profile_name=" << train.profile_name << '\n';
    std::cout << "data=" << train.data_path << '\n';
    std::cout << "checkpoint=" << train.checkpoint_path << '\n';
    std::cout << "checkpoint_inspected=true\n";
    return 0;
}

int RunPrepareKgPipelineCommand(const PrepareKgPipelineCommandOptions& options)
{
    PrepareKgCommandOptions prepare = options.prepare;
    prepare.json_output = false;

    InspectKgDatasetCommandOptions inspect_data;
    inspect_data.data_path = prepare.output_path;
    inspect_data.entity_count = options.train_kg.entity_count;
    inspect_data.relation_count = options.train_kg.relation_count;
    inspect_data.json_output = false;

    std::ostringstream sink;
    std::streambuf* old_buffer = std::cout.rdbuf(sink.rdbuf());
    try {
        RunPrepareKgCommand(prepare);
        RunInspectKgDatasetCommand(inspect_data);
        if (options.target == "kg") {
            TrainKgCommandOptions train = options.train_kg;
            train.data_path = prepare.output_path;
            train.model_name = options.model_name;
            train.json_output = false;
            RunTrainKgCommand(train);

            InspectKgCheckpointCommandOptions inspect_checkpoint;
            inspect_checkpoint.checkpoint_path = train.checkpoint_path;
            inspect_checkpoint.model_name = train.model_name;
            inspect_checkpoint.entity_count = train.entity_count;
            inspect_checkpoint.relation_count = train.relation_count;
            inspect_checkpoint.embedding_dim = train.embedding_dim;
            RunInspectKgCheckpointCommand(inspect_checkpoint);
        } else {
            TrainRgcnCommandOptions train = options.train_rgcn;
            train.data_path = prepare.output_path;
            train.json_output = false;
            RunTrainRgcnCommand(train);

            InspectRgcnCheckpointCommandOptions inspect_checkpoint;
            inspect_checkpoint.checkpoint_path = train.checkpoint_path;
            inspect_checkpoint.node_count = train.node_count;
            inspect_checkpoint.relation_count = train.relation_count;
            inspect_checkpoint.feature_dim = train.feature_dim;
            RunInspectRgcnCheckpointCommand(inspect_checkpoint);
        }
    } catch (...) {
        std::cout.rdbuf(old_buffer);
        throw;
    }
    std::cout.rdbuf(old_buffer);

    if (options.json_output) {
        const std::string checkpoint = options.target == "kg"
            ? options.train_kg.checkpoint_path
            : options.train_rgcn.checkpoint_path;
        std::cout << "{\"ok\":true,\"command\":\"prepare-kg-pipeline\",";
        WriteJsonStringField(std::cout, "profile", options.profile_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "profile_name", options.profile_name);
        std::cout << ',';
        WriteJsonStringField(std::cout, "target", options.target);
        std::cout << ',';
        WriteJsonStringField(std::cout, "input", prepare.input_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "prepared_data", prepare.output_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "checkpoint", checkpoint);
        std::cout << ",\"entities\":" << options.train_kg.entity_count
                  << ",\"relations\":" << options.train_kg.relation_count
                  << ",\"dim\":" << options.train_kg.embedding_dim
                  << ",\"prepared\":true"
                  << ",\"dataset_inspected\":true"
                  << ",\"checkpoint_inspected\":true"
                  << "}\n";
        return 0;
    }

    std::cout << "profile=" << options.profile_path << '\n';
    std::cout << "profile_name=" << options.profile_name << '\n';
    std::cout << "target=" << options.target << '\n';
    std::cout << "prepared=true\n";
    std::cout << "dataset_inspected=true\n";
    std::cout << "checkpoint_inspected=true\n";
    return 0;
}

int RunPipelineCommand(const PipelineCommandOptions& options)
{
    if (options.json_output) {
        PrepareTextCommandOptions prepare = options.prepare;
        TrainCommandOptions train = options.train;
        prepare.json_output = false;
        train.json_output = false;

        std::ostringstream sink;
        std::streambuf* old_buffer = std::cout.rdbuf(sink.rdbuf());
        try {
            RunPrepareTextCommand(prepare);
            RunTrainCommand(train);
            InspectCheckpointCommandOptions inspect;
            inspect.checkpoint_path = train.checkpoint_path;
            inspect.vocab_size = train.vocab_size;
            inspect.hidden_size = kTinyHiddenSize;
            RunInspectCheckpointCommand(inspect);
        } catch (...) {
            std::cout.rdbuf(old_buffer);
            throw;
        }
        std::cout.rdbuf(old_buffer);

        std::cout << "{\"ok\":true,\"command\":\"pipeline\",";
        if (!options.config_path.empty()) {
            WriteJsonStringField(std::cout, "config", options.config_path);
            std::cout << ',';
        }
        if (!options.profile_name.empty()) {
            WriteJsonStringField(std::cout, "profile", options.profile_path);
            std::cout << ',';
            WriteJsonStringField(std::cout, "profile_name", options.profile_name);
            std::cout << ',';
        }
        WriteJsonStringField(std::cout, "prepared_data", options.prepare.output_path);
        std::cout << ',';
        WriteJsonStringField(std::cout, "checkpoint", options.train.checkpoint_path);
        std::cout << ",\"checkpoint_inspected\":true";
        std::cout << "}\n";
        return 0;
    }

    std::cout << "[prepare]\n";
    RunPrepareTextCommand(options.prepare);
    std::cout << "[train]\n";
    RunTrainCommand(options.train);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc > 1) {
            const std::string command = argv[1];
            if (command == "--help" || command == "-h") {
                PrintUsage();
                return 0;
            }
            if (command == "train") {
                return RunTrainCommand(ParseTrainOptions(argc, argv));
            }
            if (command == "prepare-text") {
                return RunPrepareTextCommand(ParsePrepareTextOptions(argc, argv));
            }
            if (command == "pipeline") {
                return RunPipelineCommand(ParsePipelineOptions(argc, argv));
            }
            if (command == "inspect-dataset") {
                return RunInspectDatasetCommand(ParseInspectDatasetOptions(argc, argv));
            }
            if (command == "inspect-text") {
                return RunInspectTextCommand(ParseInspectTextOptions(argc, argv));
            }
            if (command == "prepare-kg") {
                return RunPrepareKgCommand(ParsePrepareKgOptions(argc, argv));
            }
            if (command == "prepare-kg-pipeline") {
                return RunPrepareKgPipelineCommand(ParsePrepareKgPipelineOptions(argc, argv));
            }
            if (command == "inspect-kg-dataset") {
                return RunInspectKgDatasetCommand(ParseInspectKgDatasetOptions(argc, argv));
            }
            if (command == "inspect-checkpoint") {
                return RunInspectCheckpointCommand(ParseInspectCheckpointOptions(argc, argv));
            }
            if (command == "generate-checkpoint") {
                return RunGenerateCheckpointCommand(ParseGenerateCheckpointOptions(argc, argv));
            }
            if (command == "validate-lm-profile") {
                return RunValidateLmProfileCommand(ParseValidateLmProfileOptions(argc, argv));
            }
            if (command == "validate-kg-profile") {
                return RunValidateKgProfileCommand(ParseValidateKgProfileOptions(argc, argv));
            }
            if (command == "validate-rgcn-profile") {
                return RunValidateRgcnProfileCommand(ParseValidateRgcnProfileOptions(argc, argv));
            }
            if (command == "train-kg") {
                return RunTrainKgCommand(ParseTrainKgOptions(argc, argv));
            }
            if (command == "inspect-kg-checkpoint") {
                return RunInspectKgCheckpointCommand(ParseInspectKgCheckpointOptions(argc, argv));
            }
            if (command == "kg-pipeline") {
                return RunKgPipelineCommand(ParseKgPipelineOptions(argc, argv));
            }
            if (command == "train-rgcn") {
                return RunTrainRgcnCommand(ParseTrainRgcnOptions(argc, argv));
            }
            if (command == "inspect-rgcn-checkpoint") {
                return RunInspectRgcnCheckpointCommand(ParseInspectRgcnCheckpointOptions(argc, argv));
            }
            if (command == "rgcn-pipeline") {
                return RunRgcnPipelineCommand(ParseRgcnPipelineOptions(argc, argv));
            }
            if (command == "generate") {
                return RunGenerateCommand(argc > 2 ? argv[2] : "1");
            }
        }
        return RunGenerateCommand(argc > 1 ? argv[1] : "1");
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
