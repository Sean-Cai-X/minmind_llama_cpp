#include <minmind/dataset.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace minmind {
namespace {

std::size_t FindKey(const std::string& text, const std::string& key)
{
    const std::string quoted_key = "\"" + key + "\"";
    const std::size_t pos = text.find(quoted_key);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing dataset key: " + key);
    }
    return pos + quoted_key.size();
}

TokenIds ExtractIntArray(const std::string& text, const std::string& key)
{
    std::size_t pos = FindKey(text, key);
    pos = text.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing array value for dataset key: " + key);
    }
    const std::size_t end = text.find(']', pos);
    if (end == std::string::npos) {
        throw std::runtime_error("unterminated array value for dataset key: " + key);
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
            throw std::runtime_error("invalid integer in dataset key: " + key);
        }
        values.push_back(std::stoi(body.substr(i, token_end - i)));
        i = token_end;
    }
    return values;
}

void ValidateSample(const TrainSample& sample)
{
    if (sample.input_ids.empty()) {
        throw std::runtime_error("dataset sample input_ids must not be empty");
    }
    if (sample.input_ids.size() != sample.labels.size()) {
        throw std::runtime_error("dataset sample input_ids and labels must have the same length");
    }
}

void WriteIntArray(std::ostream& output, const TokenIds& values)
{
    output << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << values[i];
    }
    output << ']';
}

} // namespace

std::vector<TrainSample> BuildNextTokenTrainSamples(
    const TokenIds& token_ids,
    std::size_t sequence_length,
    std::size_t max_samples)
{
    if (sequence_length == 0) {
        throw std::runtime_error("sequence_length must be greater than zero");
    }
    if (token_ids.size() <= sequence_length) {
        throw std::runtime_error("not enough token ids to build next-token samples");
    }

    std::vector<TrainSample> samples;
    const std::size_t available = (token_ids.size() - 1) / sequence_length;
    const std::size_t sample_count = max_samples == 0 ? available : std::min(available, max_samples);
    samples.reserve(sample_count);

    for (std::size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
        const std::size_t offset = sample_index * sequence_length;
        TrainSample sample;
        sample.input_ids.reserve(sequence_length);
        sample.labels.reserve(sequence_length);
        for (std::size_t i = 0; i < sequence_length; ++i) {
            sample.input_ids.push_back(token_ids[offset + i]);
            sample.labels.push_back(token_ids[offset + i + 1]);
        }
        samples.push_back(std::move(sample));
    }

    if (samples.empty()) {
        throw std::runtime_error("next-token sample builder produced no samples");
    }
    return samples;
}

std::vector<TrainSample> LoadJsonlTrainSamples(const std::string& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open dataset: " + path);
    }

    std::vector<TrainSample> samples;
    std::string line;
    while (std::getline(input, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        TrainSample sample;
        sample.input_ids = ExtractIntArray(line, "input_ids");
        sample.labels = ExtractIntArray(line, "labels");
        ValidateSample(sample);
        samples.push_back(std::move(sample));
    }

    if (samples.empty()) {
        throw std::runtime_error("dataset must contain at least one sample");
    }
    return samples;
}

void SaveJsonlTrainSamples(const std::string& path, const std::vector<TrainSample>& samples)
{
    if (samples.empty()) {
        throw std::runtime_error("dataset must contain at least one sample");
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open dataset for writing: " + path);
    }

    for (const TrainSample& sample : samples) {
        ValidateSample(sample);
        output << "{\"input_ids\":";
        WriteIntArray(output, sample.input_ids);
        output << ",\"labels\":";
        WriteIntArray(output, sample.labels);
        output << "}\n";
    }
}

} // namespace minmind
