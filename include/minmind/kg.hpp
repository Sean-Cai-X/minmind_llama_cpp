#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace minmind {

struct KgTriple {
    std::size_t head = 0;
    std::size_t relation = 0;
    std::size_t tail = 0;
};

enum class KgEmbeddingModel {
    TransE,
    DistMult,
    RotatE
};

struct KgEmbeddingConfig {
    KgEmbeddingModel model = KgEmbeddingModel::TransE;
    std::size_t entity_count = 0;
    std::size_t relation_count = 0;
    std::size_t embedding_dim = 8;
    std::size_t epochs = 4;
    float learning_rate = 0.01f;
    float margin = 1.0f;
};

struct KgEmbeddingState {
    std::vector<float> entity_embeddings;
    std::vector<float> relation_embeddings;
    std::size_t entity_width = 0;
    std::size_t relation_width = 0;
};

struct KgTrainResult {
    std::vector<float> loss_history;
};

struct RgcnTrainConfig {
    std::size_t node_count = 0;
    std::size_t relation_count = 0;
    std::size_t feature_dim = 8;
    std::size_t epochs = 4;
    float learning_rate = 0.01f;
};

struct RgcnTrainResult {
    std::vector<float> loss_history;
};

KgEmbeddingModel ParseKgEmbeddingModel(const std::string& name);
const char* KgEmbeddingModelName(KgEmbeddingModel model);

std::vector<KgTriple> LoadKgTriplesTsv(const std::string& path);
KgEmbeddingState InitKgEmbeddingState(const KgEmbeddingConfig& config);
float ScoreKgTriple(const KgEmbeddingState& state, KgEmbeddingModel model, const KgTriple& triple);
KgTrainResult TrainKgEmbeddings(
    const std::vector<KgTriple>& triples,
    const KgEmbeddingConfig& config,
    KgEmbeddingState& state);

void SaveKgEmbeddingCheckpoint(
    const std::string& path,
    KgEmbeddingModel model,
    std::size_t entity_count,
    std::size_t relation_count,
    std::size_t embedding_dim,
    const KgEmbeddingState& state);

void LoadKgEmbeddingCheckpoint(
    const std::string& path,
    KgEmbeddingModel expected_model,
    std::size_t expected_entity_count,
    std::size_t expected_relation_count,
    std::size_t expected_embedding_dim,
    KgEmbeddingState& state);

std::vector<float> RgcnRelationMeanForward(
    const std::vector<KgTriple>& triples,
    const std::vector<float>& node_features,
    const std::vector<float>& relation_weights,
    std::size_t node_count,
    std::size_t relation_count,
    std::size_t feature_dim);

std::vector<float> InitRgcnNodeFeatures(std::size_t node_count, std::size_t feature_dim);
std::vector<float> InitRgcnRelationWeights(std::size_t relation_count, std::size_t feature_dim);
RgcnTrainResult TrainRgcnRelationMean(
    const std::vector<KgTriple>& triples,
    const RgcnTrainConfig& config,
    std::vector<float>& node_features,
    std::vector<float>& relation_weights);

void SaveRgcnRelationCheckpoint(
    const std::string& path,
    std::size_t node_count,
    std::size_t relation_count,
    std::size_t feature_dim,
    const std::vector<float>& relation_weights);

void LoadRgcnRelationCheckpoint(
    const std::string& path,
    std::size_t expected_node_count,
    std::size_t expected_relation_count,
    std::size_t expected_feature_dim,
    std::vector<float>& relation_weights);

} // namespace minmind
