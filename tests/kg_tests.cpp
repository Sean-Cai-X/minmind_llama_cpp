#include "test_assert.hpp"

#include <minmind/kg.hpp>

#include <string>
#include <vector>

namespace minmind::tests {

void KgEmbeddingTrainingProducesLossForAllModels()
{
    const std::vector<KgTriple> triples = {
        {0, 0, 1},
        {1, 0, 2},
        {2, 1, 3},
    };

    const std::vector<KgEmbeddingModel> models = {
        KgEmbeddingModel::TransE,
        KgEmbeddingModel::DistMult,
        KgEmbeddingModel::RotatE,
    };
    for (const KgEmbeddingModel model : models) {
        KgEmbeddingConfig config;
        config.model = model;
        config.entity_count = 4;
        config.relation_count = 2;
        config.embedding_dim = 3;
        config.epochs = 2;
        config.learning_rate = 0.01f;

        KgEmbeddingState state = InitKgEmbeddingState(config);
        const KgTrainResult result = TrainKgEmbeddings(triples, config, state);

        MINMIND_REQUIRE(result.loss_history.size() == 2);
        MINMIND_REQUIRE(result.loss_history[0] >= 0.0f);
        MINMIND_REQUIRE(!state.entity_embeddings.empty());
        MINMIND_REQUIRE(!state.relation_embeddings.empty());
    }
}

void RgcnRelationMeanForwardAggregatesIncomingEdges()
{
    const std::vector<KgTriple> triples = {
        {0, 0, 1},
        {2, 1, 1},
    };
    const std::vector<float> node_features = {
        1.0f, 2.0f,
        0.0f, 0.0f,
        3.0f, 4.0f,
    };
    const std::vector<float> relation_weights = {
        2.0f, 3.0f,
        4.0f, 5.0f,
    };

    const std::vector<float> output = RgcnRelationMeanForward(
        triples,
        node_features,
        relation_weights,
        3,
        2,
        2);

    MINMIND_REQUIRE(output.size() == 6);
    MINMIND_REQUIRE(output[2] == 7.0f);
    MINMIND_REQUIRE(output[3] == 13.0f);
}

void KgEmbeddingCheckpointReloadsShapeAndWeights()
{
    KgEmbeddingConfig config;
    config.model = KgEmbeddingModel::DistMult;
    config.entity_count = 3;
    config.relation_count = 2;
    config.embedding_dim = 2;

    KgEmbeddingState state = InitKgEmbeddingState(config);
    const std::string path = "minmind_kg_embedding_checkpoint.tmp";
    SaveKgEmbeddingCheckpoint(
        path,
        config.model,
        config.entity_count,
        config.relation_count,
        config.embedding_dim,
        state);

    KgEmbeddingState loaded;
    LoadKgEmbeddingCheckpoint(
        path,
        config.model,
        config.entity_count,
        config.relation_count,
        config.embedding_dim,
        loaded);

    MINMIND_REQUIRE(loaded.entity_width == state.entity_width);
    MINMIND_REQUIRE(loaded.relation_width == state.relation_width);
    MINMIND_REQUIRE(loaded.entity_embeddings == state.entity_embeddings);
    MINMIND_REQUIRE(loaded.relation_embeddings == state.relation_embeddings);
}

void RgcnRelationMeanTrainingProducesLossHistory()
{
    const std::vector<KgTriple> triples = {
        {0, 0, 1},
        {1, 0, 2},
        {2, 1, 3},
    };

    RgcnTrainConfig config;
    config.node_count = 4;
    config.relation_count = 2;
    config.feature_dim = 3;
    config.epochs = 2;
    config.learning_rate = 0.01f;

    std::vector<float> node_features = InitRgcnNodeFeatures(config.node_count, config.feature_dim);
    std::vector<float> relation_weights = InitRgcnRelationWeights(config.relation_count, config.feature_dim);
    const RgcnTrainResult result = TrainRgcnRelationMean(
        triples,
        config,
        node_features,
        relation_weights);

    MINMIND_REQUIRE(result.loss_history.size() == 2);
    MINMIND_REQUIRE(result.loss_history[0] >= 0.0f);
    MINMIND_REQUIRE(relation_weights.size() == config.relation_count * config.feature_dim);
}

void RgcnRelationCheckpointReloadsWeights()
{
    const std::size_t node_count = 4;
    const std::size_t relation_count = 2;
    const std::size_t feature_dim = 3;
    const std::string path = "minmind_rgcn_relation_checkpoint.tmp";
    const std::vector<float> relation_weights = InitRgcnRelationWeights(relation_count, feature_dim);

    SaveRgcnRelationCheckpoint(
        path,
        node_count,
        relation_count,
        feature_dim,
        relation_weights);

    std::vector<float> loaded;
    LoadRgcnRelationCheckpoint(
        path,
        node_count,
        relation_count,
        feature_dim,
        loaded);

    MINMIND_REQUIRE(loaded == relation_weights);
}

} // namespace minmind::tests
