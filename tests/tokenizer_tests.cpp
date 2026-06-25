#include "test_assert.hpp"

#include <minmind/tokenizer.hpp>

namespace minmind::tests {

void TokenIdsRoundTrip()
{
    const TokenIds ids = ParseTokenIds("1 2 3 42");
    MINMIND_REQUIRE(ids.size() == 4);
    MINMIND_REQUIRE(ids[0] == 1);
    MINMIND_REQUIRE(ids[3] == 42);
    MINMIND_REQUIRE(JoinTokenIds(ids) == "1 2 3 42");
}

void Vocab4CharTokenizerMapsText()
{
    const TokenIds ids = EncodeVocab4Chars("Am Nz!\n");
    MINMIND_REQUIRE(ids == TokenIds({1, 1, 0, 2, 2, 3, 0}));
}

void ByteVocabTokenizerUsesTopBytesAndUnknown()
{
    const ByteVocab vocab = BuildByteVocab("aaabbc", 3);
    MINMIND_REQUIRE(vocab.unknown_id == 0);
    MINMIND_REQUIRE(vocab.id_to_byte.size() == 2);
    MINMIND_REQUIRE(vocab.id_to_byte[0] == static_cast<unsigned char>('a'));
    MINMIND_REQUIRE(vocab.id_to_byte[1] == static_cast<unsigned char>('b'));

    const TokenIds ids = EncodeBytesWithVocab("abc", vocab);
    MINMIND_REQUIRE(ids == TokenIds({1, 2, 0}));
}

} // namespace minmind::tests
