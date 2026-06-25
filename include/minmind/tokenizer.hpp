#pragma once

#include <minmind/model.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace minmind {

struct ByteVocab {
    int unknown_id = 0;
    std::vector<unsigned char> id_to_byte;
};

// Phase 1 tokenizer shim. Phase 2 will replace this with golden fixture loading
// and, later, tokenizer.json/BPE support.
TokenIds ParseTokenIds(const std::string& text);
std::string JoinTokenIds(const TokenIds& token_ids);
TokenIds EncodeVocab4Chars(const std::string& text);
ByteVocab BuildByteVocab(const std::string& text, std::size_t vocab_size);
TokenIds EncodeBytesWithVocab(const std::string& text, const ByteVocab& vocab);
void SaveByteVocab(const std::string& path, const ByteVocab& vocab);
ByteVocab LoadByteVocab(const std::string& path);

} // namespace minmind
