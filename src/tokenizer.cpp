#include <minmind/tokenizer.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace minmind {
namespace {

constexpr const char* kByteVocabMagic = "MINMIND_BYTE_VOCAB_V1";

int EncodeVocab4Char(unsigned char ch)
{
    if (std::isspace(ch)) {
        return 0;
    }

    const unsigned char lower = static_cast<unsigned char>(std::tolower(ch));
    if (lower >= 'a' && lower <= 'm') {
        return 1;
    }
    if (lower >= 'n' && lower <= 'z') {
        return 2;
    }
    return 3;
}

void ValidateByteVocab(const ByteVocab& vocab)
{
    if (vocab.unknown_id != 0) {
        throw std::invalid_argument("byte vocab unknown_id must be 0");
    }
}

} // namespace

TokenIds ParseTokenIds(const std::string& text)
{
    TokenIds ids;
    std::istringstream input(text);
    int id = 0;
    while (input >> id) {
        ids.push_back(id);
    }
    return ids;
}

std::string JoinTokenIds(const TokenIds& token_ids)
{
    std::ostringstream output;
    for (std::size_t i = 0; i < token_ids.size(); ++i) {
        if (i != 0) {
            output << ' ';
        }
        output << token_ids[i];
    }
    return output.str();
}

TokenIds EncodeVocab4Chars(const std::string& text)
{
    TokenIds ids;
    ids.reserve(text.size());
    for (const unsigned char ch : text) {
        ids.push_back(EncodeVocab4Char(ch));
    }
    return ids;
}

ByteVocab BuildByteVocab(const std::string& text, std::size_t vocab_size)
{
    if (vocab_size < 2 || vocab_size > 256) {
        throw std::invalid_argument("byte vocab_size must be in [2, 256]");
    }

    std::array<std::size_t, 256> counts{};
    for (const unsigned char ch : text) {
        ++counts[ch];
    }

    std::vector<unsigned char> bytes;
    bytes.reserve(256);
    for (std::size_t value = 0; value < counts.size(); ++value) {
        if (counts[value] > 0) {
            bytes.push_back(static_cast<unsigned char>(value));
        }
    }

    std::sort(bytes.begin(), bytes.end(), [&counts](unsigned char lhs, unsigned char rhs) {
        if (counts[lhs] != counts[rhs]) {
            return counts[lhs] > counts[rhs];
        }
        return lhs < rhs;
    });

    const std::size_t kept_byte_count = std::min(bytes.size(), vocab_size - 1);
    ByteVocab vocab;
    vocab.id_to_byte.assign(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(kept_byte_count));
    return vocab;
}

TokenIds EncodeBytesWithVocab(const std::string& text, const ByteVocab& vocab)
{
    ValidateByteVocab(vocab);

    std::array<int, 256> byte_to_id{};
    for (std::size_t i = 0; i < byte_to_id.size(); ++i) {
        byte_to_id[i] = vocab.unknown_id;
    }
    for (std::size_t i = 0; i < vocab.id_to_byte.size(); ++i) {
        byte_to_id[vocab.id_to_byte[i]] = static_cast<int>(i + 1);
    }

    TokenIds ids;
    ids.reserve(text.size());
    for (const unsigned char ch : text) {
        ids.push_back(byte_to_id[ch]);
    }
    return ids;
}

void SaveByteVocab(const std::string& path, const ByteVocab& vocab)
{
    ValidateByteVocab(vocab);

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open byte vocab for writing: " + path);
    }

    output << kByteVocabMagic << '\n';
    output << "unknown " << vocab.unknown_id << '\n';
    for (const unsigned char ch : vocab.id_to_byte) {
        output << "byte " << static_cast<int>(ch) << '\n';
    }
}

ByteVocab LoadByteVocab(const std::string& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open byte vocab: " + path);
    }

    std::string magic;
    input >> magic;
    if (magic != kByteVocabMagic) {
        throw std::runtime_error("invalid byte vocab magic");
    }

    std::string label;
    ByteVocab vocab;
    input >> label >> vocab.unknown_id;
    if (label != "unknown") {
        throw std::runtime_error("byte vocab missing unknown row");
    }

    int value = 0;
    while (input >> label >> value) {
        if (label != "byte") {
            throw std::runtime_error("invalid byte vocab row: " + label);
        }
        if (value < 0 || value > 255) {
            throw std::runtime_error("byte vocab row outside byte range");
        }
        vocab.id_to_byte.push_back(static_cast<unsigned char>(value));
    }

    ValidateByteVocab(vocab);
    return vocab;
}

} // namespace minmind
