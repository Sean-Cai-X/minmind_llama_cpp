#include "test_assert.hpp"

namespace minmind::tests {

void CausalSelfAttentionMatchesGoldenFixture();
void CausalLmForwardLossMatchesGoldenFixture();
void CausalLmGreedyGenerationMatchesGoldenFixture();
void CausalLmHeadCheckpointReloadReproducesLoss();
void CausalLmHeadOnlyTrainLoopReducesLossHistory();
void CausalLmHeadOnlyTrainStepReducesLoss();
void CausalLmMatchesGoldenFixture();
void ConfigDefaultsAreValid();
void JsonlTrainSamplesDriveTinyTrainLoop();
void JsonlTrainSamplesLoadTokenIdsAndLabels();
void NextTokenTrainSamplesUseFixedLengthWindows();
void TransformerBlockMatchesGoldenFixture();
void InvalidConfigReportsError();
void KgEmbeddingTrainingProducesLossForAllModels();
void KgEmbeddingCheckpointReloadsShapeAndWeights();
void GatedFeedForwardMatchesGoldenFixture();
void GoldenForwardFixtureLoadsLossAndLogits();
void GoldenFixtureLoadsTokenIds();
void MeanCrossEntropyLossMatchesGoldenFixture();
void McpToolManifestIncludesCheckpointTrainingTools();
void RgcnRelationMeanForwardAggregatesIncomingEdges();
void RgcnRelationMeanTrainingProducesLossHistory();
void RgcnRelationCheckpointReloadsWeights();
void TokenIdsRoundTrip();
void Vocab4CharTokenizerMapsText();
void ByteVocabTokenizerUsesTopBytesAndUnknown();
void PlaceholderForwardChoosesNextToken();
void RopeMatchesGoldenFixture();
void RmsNormMatchesGoldenFixture();
void GreedySampleChoosesLargestLogit();

int RunTest(const char* name, TestFn test)
{
    try {
        test();
        std::cout << "[pass] " << name << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[fail] " << name << ": " << ex.what() << '\n';
        return 1;
    }
}

} // namespace minmind::tests

int main()
{
    int failures = 0;
    failures += minmind::tests::RunTest("CausalSelfAttentionMatchesGoldenFixture", minmind::tests::CausalSelfAttentionMatchesGoldenFixture);
    failures += minmind::tests::RunTest("CausalLmForwardLossMatchesGoldenFixture", minmind::tests::CausalLmForwardLossMatchesGoldenFixture);
    failures += minmind::tests::RunTest("CausalLmGreedyGenerationMatchesGoldenFixture", minmind::tests::CausalLmGreedyGenerationMatchesGoldenFixture);
    failures += minmind::tests::RunTest("CausalLmHeadCheckpointReloadReproducesLoss", minmind::tests::CausalLmHeadCheckpointReloadReproducesLoss);
    failures += minmind::tests::RunTest("CausalLmHeadOnlyTrainLoopReducesLossHistory", minmind::tests::CausalLmHeadOnlyTrainLoopReducesLossHistory);
    failures += minmind::tests::RunTest("CausalLmHeadOnlyTrainStepReducesLoss", minmind::tests::CausalLmHeadOnlyTrainStepReducesLoss);
    failures += minmind::tests::RunTest("CausalLmMatchesGoldenFixture", minmind::tests::CausalLmMatchesGoldenFixture);
    failures += minmind::tests::RunTest("ConfigDefaultsAreValid", minmind::tests::ConfigDefaultsAreValid);
    failures += minmind::tests::RunTest("JsonlTrainSamplesDriveTinyTrainLoop", minmind::tests::JsonlTrainSamplesDriveTinyTrainLoop);
    failures += minmind::tests::RunTest("JsonlTrainSamplesLoadTokenIdsAndLabels", minmind::tests::JsonlTrainSamplesLoadTokenIdsAndLabels);
    failures += minmind::tests::RunTest("NextTokenTrainSamplesUseFixedLengthWindows", minmind::tests::NextTokenTrainSamplesUseFixedLengthWindows);
    failures += minmind::tests::RunTest("TransformerBlockMatchesGoldenFixture", minmind::tests::TransformerBlockMatchesGoldenFixture);
    failures += minmind::tests::RunTest("InvalidConfigReportsError", minmind::tests::InvalidConfigReportsError);
    failures += minmind::tests::RunTest("KgEmbeddingTrainingProducesLossForAllModels", minmind::tests::KgEmbeddingTrainingProducesLossForAllModels);
    failures += minmind::tests::RunTest("KgEmbeddingCheckpointReloadsShapeAndWeights", minmind::tests::KgEmbeddingCheckpointReloadsShapeAndWeights);
    failures += minmind::tests::RunTest("GatedFeedForwardMatchesGoldenFixture", minmind::tests::GatedFeedForwardMatchesGoldenFixture);
    failures += minmind::tests::RunTest("GoldenForwardFixtureLoadsLossAndLogits", minmind::tests::GoldenForwardFixtureLoadsLossAndLogits);
    failures += minmind::tests::RunTest("GoldenFixtureLoadsTokenIds", minmind::tests::GoldenFixtureLoadsTokenIds);
    failures += minmind::tests::RunTest("MeanCrossEntropyLossMatchesGoldenFixture", minmind::tests::MeanCrossEntropyLossMatchesGoldenFixture);
    failures += minmind::tests::RunTest("McpToolManifestIncludesCheckpointTrainingTools", minmind::tests::McpToolManifestIncludesCheckpointTrainingTools);
    failures += minmind::tests::RunTest("RgcnRelationMeanForwardAggregatesIncomingEdges", minmind::tests::RgcnRelationMeanForwardAggregatesIncomingEdges);
    failures += minmind::tests::RunTest("RgcnRelationMeanTrainingProducesLossHistory", minmind::tests::RgcnRelationMeanTrainingProducesLossHistory);
    failures += minmind::tests::RunTest("RgcnRelationCheckpointReloadsWeights", minmind::tests::RgcnRelationCheckpointReloadsWeights);
    failures += minmind::tests::RunTest("TokenIdsRoundTrip", minmind::tests::TokenIdsRoundTrip);
    failures += minmind::tests::RunTest("Vocab4CharTokenizerMapsText", minmind::tests::Vocab4CharTokenizerMapsText);
    failures += minmind::tests::RunTest("ByteVocabTokenizerUsesTopBytesAndUnknown", minmind::tests::ByteVocabTokenizerUsesTopBytesAndUnknown);
    failures += minmind::tests::RunTest("PlaceholderForwardChoosesNextToken", minmind::tests::PlaceholderForwardChoosesNextToken);
    failures += minmind::tests::RunTest("RopeMatchesGoldenFixture", minmind::tests::RopeMatchesGoldenFixture);
    failures += minmind::tests::RunTest("RmsNormMatchesGoldenFixture", minmind::tests::RmsNormMatchesGoldenFixture);
    failures += minmind::tests::RunTest("GreedySampleChoosesLargestLogit", minmind::tests::GreedySampleChoosesLargestLogit);
    return failures == 0 ? 0 : 1;
}
