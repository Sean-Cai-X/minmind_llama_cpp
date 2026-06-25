#include <minmind/kg.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace minmind {
namespace {

constexpr const char* kKgEmbeddingCheckpointMagic = "MINMIND_KG_EMBEDDING_V1";
constexpr const char* kRgcnRelationCheckpointMagic = "MINMIND_RGCN_RELATION_V1";

std::size_t EntityOffset(const KgEmbeddingState& state, std::size_t entity)
{
    return entity * state.entity_width;
}

std::size_t RelationOffset(const KgEmbeddingState& state, std::size_t relation)
{
    return relation * state.relation_width;
}

void ValidateTriple(const KgTriple& triple, const KgEmbeddingConfig& config)
{
    if (triple.head >= config.entity_count || triple.tail >= config.entity_count) {
        throw std::runtime_error("KG triple entity id is outside entity_count");
    }
    if (triple.relation >= config.relation_count) {
        throw std::runtime_error("KG triple relation id is outside relation_count");
    }
}

float MarginLossForTriple(
    const KgEmbeddingState& state,
    KgEmbeddingModel model,
    const KgTriple& positive,
    std::size_t entity_count,
    float margin)
{
    KgTriple negative = positive;
    negative.tail = entity_count <= 1 ? positive.tail : (positive.tail + 1) % entity_count;
    const float positive_score = ScoreKgTriple(state, model, positive);
    const float negative_score = ScoreKgTriple(state, model, negative);
    return std::max(0.0f, margin - positive_score + negative_score);
}

void FiniteDifferenceUpdate(
    float* value,
    const KgEmbeddingState& state,
    KgEmbeddingModel model,
    const KgTriple& triple,
    std::size_t entity_count,
    float margin,
    float learning_rate)
{
    constexpr float kEps = 0.001f;
    const float original = *value;

    *value = original + kEps;
    const float plus_loss = MarginLossForTriple(state, model, triple, entity_count, margin);

    *value = original - kEps;
    const float minus_loss = MarginLossForTriple(state, model, triple, entity_count, margin);

    *value = original;
    const float grad = (plus_loss - minus_loss) / (2.0f * kEps);
    *value -= learning_rate * grad;
}

float RgcnReconstructionLoss(
    const std::vector<KgTriple>& triples,
    const std::vector<float>& node_features,
    const std::vector<float>& relation_weights,
    std::size_t node_count,
    std::size_t relation_count,
    std::size_t feature_dim)
{
    const std::vector<float> output = RgcnRelationMeanForward(
        triples,
        node_features,
        relation_weights,
        node_count,
        relation_count,
        feature_dim);

    float loss = 0.0f;
    std::size_t terms = 0;
    for (const KgTriple& triple : triples) {
        for (std::size_t dim = 0; dim < feature_dim; ++dim) {
            const float diff = output[triple.tail * feature_dim + dim] -
                node_features[triple.tail * feature_dim + dim];
            loss += diff * diff;
            ++terms;
        }
    }
    return terms == 0 ? 0.0f : loss / static_cast<float>(terms);
}

} // namespace

KgEmbeddingModel ParseKgEmbeddingModel(const std::string& name)
{
    if (name == "transe") {
        return KgEmbeddingModel::TransE;
    }
    if (name == "distmult") {
        return KgEmbeddingModel::DistMult;
    }
    if (name == "rotate") {
        return KgEmbeddingModel::RotatE;
    }
    throw std::invalid_argument("unknown KG embedding model: " + name);
}

const char* KgEmbeddingModelName(KgEmbeddingModel model)
{
    switch (model) {
    case KgEmbeddingModel::TransE:
        return "transe";
    case KgEmbeddingModel::DistMult:
        return "distmult";
    case KgEmbeddingModel::RotatE:
        return "rotate";
    }
    return "unknown";
}

std::vector<KgTriple> LoadKgTriplesTsv(const std::string& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open KG triples: " + path);
    }

    std::vector<KgTriple> triples;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream parsed(line);
        KgTriple triple;
        parsed >> triple.head >> triple.relation >> triple.tail;
        if (!parsed) {
            throw std::runtime_error("invalid KG triple line: " + line);
        }
        triples.push_back(triple);
    }
    if (triples.empty()) {
        throw std::runtime_error("KG triples file is empty");
    }
    return triples;
}

KgEmbeddingState InitKgEmbeddingState(const KgEmbeddingConfig& config)
{
    if (config.entity_count == 0 || config.relation_count == 0 || config.embedding_dim == 0) {
        throw std::invalid_argument("KG embedding dimensions must be greater than zero");
    }

    KgEmbeddingState state;
    state.entity_width = config.model == KgEmbeddingModel::RotatE
        ? config.embedding_dim * 2
        : config.embedding_dim;
    state.relation_width = config.embedding_dim;
    state.entity_embeddings.resize(config.entity_count * state.entity_width, 0.0f);
    state.relation_embeddings.resize(config.relation_count * state.relation_width, 0.0f);

    for (std::size_t i = 0; i < state.entity_embeddings.size(); ++i) {
        state.entity_embeddings[i] = 0.05f * static_cast<float>((i % 7) + 1);
    }
    for (std::size_t i = 0; i < state.relation_embeddings.size(); ++i) {
        state.relation_embeddings[i] = 0.03f * static_cast<float>((i % 5) + 1);
    }
    return state;
}

float ScoreKgTriple(const KgEmbeddingState& state, KgEmbeddingModel model, const KgTriple& triple)
{
    const std::size_t h = EntityOffset(state, triple.head);
    const std::size_t r = RelationOffset(state, triple.relation);
    const std::size_t t = EntityOffset(state, triple.tail);

    if (model == KgEmbeddingModel::TransE) {
        float distance = 0.0f;
        for (std::size_t dim = 0; dim < state.relation_width; ++dim) {
            distance += std::fabs(
                state.entity_embeddings[h + dim] +
                state.relation_embeddings[r + dim] -
                state.entity_embeddings[t + dim]);
        }
        return -distance;
    }

    if (model == KgEmbeddingModel::DistMult) {
        float score = 0.0f;
        for (std::size_t dim = 0; dim < state.relation_width; ++dim) {
            score += state.entity_embeddings[h + dim] *
                state.relation_embeddings[r + dim] *
                state.entity_embeddings[t + dim];
        }
        return score;
    }

    float distance = 0.0f;
    for (std::size_t dim = 0; dim < state.relation_width; ++dim) {
        const float angle = state.relation_embeddings[r + dim];
        const float cos_r = std::cos(angle);
        const float sin_r = std::sin(angle);
        const float h_re = state.entity_embeddings[h + dim * 2];
        const float h_im = state.entity_embeddings[h + dim * 2 + 1];
        const float rotated_re = h_re * cos_r - h_im * sin_r;
        const float rotated_im = h_re * sin_r + h_im * cos_r;
        const float t_re = state.entity_embeddings[t + dim * 2];
        const float t_im = state.entity_embeddings[t + dim * 2 + 1];
        distance += std::fabs(rotated_re - t_re) + std::fabs(rotated_im - t_im);
    }
    return -distance;
}

KgTrainResult TrainKgEmbeddings(
    const std::vector<KgTriple>& triples,
    const KgEmbeddingConfig& config,
    KgEmbeddingState& state)
{
    for (const KgTriple& triple : triples) {
        ValidateTriple(triple, config);
    }

    KgTrainResult result;
    result.loss_history.reserve(config.epochs);
    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        float total_loss = 0.0f;
        for (const KgTriple& triple : triples) {
            total_loss += MarginLossForTriple(state, config.model, triple, config.entity_count, config.margin);
            const std::size_t neg_tail = config.entity_count <= 1 ? triple.tail : (triple.tail + 1) % config.entity_count;
            const std::size_t entity_ids[] = {triple.head, triple.tail, neg_tail};
            for (const std::size_t entity : entity_ids) {
                const std::size_t offset = EntityOffset(state, entity);
                for (std::size_t dim = 0; dim < state.entity_width; ++dim) {
                    FiniteDifferenceUpdate(
                        &state.entity_embeddings[offset + dim],
                        state,
                        config.model,
                        triple,
                        config.entity_count,
                        config.margin,
                        config.learning_rate);
                }
            }
            const std::size_t relation_offset = RelationOffset(state, triple.relation);
            for (std::size_t dim = 0; dim < state.relation_width; ++dim) {
                FiniteDifferenceUpdate(
                    &state.relation_embeddings[relation_offset + dim],
                    state,
                    config.model,
                    triple,
                    config.entity_count,
                    config.margin,
                    config.learning_rate);
            }
        }
        result.loss_history.push_back(total_loss / static_cast<float>(triples.size()));
    }
    return result;
}

void SaveKgEmbeddingCheckpoint(
    const std::string& path,
    KgEmbeddingModel model,
    std::size_t entity_count,
    std::size_t relation_count,
    std::size_t embedding_dim,
    const KgEmbeddingState& state)
{
    if (entity_count == 0 || relation_count == 0 || embedding_dim == 0) {
        throw std::invalid_argument("KG checkpoint dimensions must be greater than zero");
    }
    if (state.entity_embeddings.size() != entity_count * state.entity_width) {
        throw std::invalid_argument("KG entity embedding size does not match checkpoint shape");
    }
    if (state.relation_embeddings.size() != relation_count * state.relation_width) {
        throw std::invalid_argument("KG relation embedding size does not match checkpoint shape");
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open KG checkpoint for writing: " + path);
    }

    output << kKgEmbeddingCheckpointMagic << '\n';
    output << KgEmbeddingModelName(model) << '\n';
    output << entity_count << ' ' << relation_count << ' ' << embedding_dim << '\n';
    output << state.entity_width << ' ' << state.relation_width << '\n';
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t i = 0; i < state.entity_embeddings.size(); ++i) {
        if (i != 0) {
            output << ' ';
        }
        output << state.entity_embeddings[i];
    }
    output << '\n';
    for (std::size_t i = 0; i < state.relation_embeddings.size(); ++i) {
        if (i != 0) {
            output << ' ';
        }
        output << state.relation_embeddings[i];
    }
    output << '\n';

    if (!output) {
        throw std::runtime_error("failed to write KG checkpoint: " + path);
    }
}

void LoadKgEmbeddingCheckpoint(
    const std::string& path,
    KgEmbeddingModel expected_model,
    std::size_t expected_entity_count,
    std::size_t expected_relation_count,
    std::size_t expected_embedding_dim,
    KgEmbeddingState& state)
{
    if (expected_entity_count == 0 || expected_relation_count == 0 || expected_embedding_dim == 0) {
        throw std::invalid_argument("KG checkpoint dimensions must be greater than zero");
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open KG checkpoint for reading: " + path);
    }

    std::string magic;
    input >> magic;
    if (magic != kKgEmbeddingCheckpointMagic) {
        throw std::runtime_error("invalid KG checkpoint magic");
    }

    std::string model_name;
    input >> model_name;
    if (ParseKgEmbeddingModel(model_name) != expected_model) {
        throw std::runtime_error("KG checkpoint model does not match requested model");
    }

    std::size_t entity_count = 0;
    std::size_t relation_count = 0;
    std::size_t embedding_dim = 0;
    std::size_t entity_width = 0;
    std::size_t relation_width = 0;
    input >> entity_count >> relation_count >> embedding_dim;
    input >> entity_width >> relation_width;

    if (entity_count != expected_entity_count ||
        relation_count != expected_relation_count ||
        embedding_dim != expected_embedding_dim) {
        throw std::runtime_error("KG checkpoint shape does not match requested shape");
    }

    state.entity_width = entity_width;
    state.relation_width = relation_width;
    state.entity_embeddings.assign(entity_count * entity_width, 0.0f);
    state.relation_embeddings.assign(relation_count * relation_width, 0.0f);
    for (float& value : state.entity_embeddings) {
        input >> value;
    }
    for (float& value : state.relation_embeddings) {
        input >> value;
    }
    if (!input) {
        throw std::runtime_error("KG checkpoint ended before all embeddings were read");
    }
}

std::vector<float> RgcnRelationMeanForward(
    const std::vector<KgTriple>& triples,
    const std::vector<float>& node_features,
    const std::vector<float>& relation_weights,
    std::size_t node_count,
    std::size_t relation_count,
    std::size_t feature_dim)
{
    if (node_features.size() != node_count * feature_dim) {
        throw std::invalid_argument("R-GCN node feature shape is invalid");
    }
    if (relation_weights.size() != relation_count * feature_dim) {
        throw std::invalid_argument("R-GCN relation weight shape is invalid");
    }

    std::vector<float> output(node_count * feature_dim, 0.0f);
    std::vector<std::size_t> degree(node_count, 0);
    for (const KgTriple& triple : triples) {
        if (triple.head >= node_count || triple.tail >= node_count || triple.relation >= relation_count) {
            throw std::runtime_error("R-GCN triple id is outside configured shape");
        }
        ++degree[triple.tail];
        for (std::size_t dim = 0; dim < feature_dim; ++dim) {
            output[triple.tail * feature_dim + dim] +=
                node_features[triple.head * feature_dim + dim] *
                relation_weights[triple.relation * feature_dim + dim];
        }
    }
    for (std::size_t node = 0; node < node_count; ++node) {
        if (degree[node] == 0) {
            continue;
        }
        for (std::size_t dim = 0; dim < feature_dim; ++dim) {
            output[node * feature_dim + dim] /= static_cast<float>(degree[node]);
        }
    }
    return output;
}

std::vector<float> InitRgcnNodeFeatures(std::size_t node_count, std::size_t feature_dim)
{
    if (node_count == 0 || feature_dim == 0) {
        throw std::invalid_argument("R-GCN node feature dimensions must be greater than zero");
    }
    std::vector<float> values(node_count * feature_dim, 0.0f);
    for (std::size_t node = 0; node < node_count; ++node) {
        for (std::size_t dim = 0; dim < feature_dim; ++dim) {
            values[node * feature_dim + dim] =
                dim == (node % feature_dim) ? 1.0f : 0.1f * static_cast<float>((node + dim) % 3);
        }
    }
    return values;
}

std::vector<float> InitRgcnRelationWeights(std::size_t relation_count, std::size_t feature_dim)
{
    if (relation_count == 0 || feature_dim == 0) {
        throw std::invalid_argument("R-GCN relation weight dimensions must be greater than zero");
    }
    std::vector<float> values(relation_count * feature_dim, 0.0f);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.2f + 0.03f * static_cast<float>(i % 5);
    }
    return values;
}

RgcnTrainResult TrainRgcnRelationMean(
    const std::vector<KgTriple>& triples,
    const RgcnTrainConfig& config,
    std::vector<float>& node_features,
    std::vector<float>& relation_weights)
{
    if (triples.empty()) {
        throw std::invalid_argument("R-GCN training requires at least one triple");
    }
    if (config.node_count == 0 || config.relation_count == 0 || config.feature_dim == 0) {
        throw std::invalid_argument("R-GCN training dimensions must be greater than zero");
    }
    for (const KgTriple& triple : triples) {
        if (triple.head >= config.node_count || triple.tail >= config.node_count || triple.relation >= config.relation_count) {
            throw std::runtime_error("R-GCN training triple id is outside configured shape");
        }
    }
    if (node_features.size() != config.node_count * config.feature_dim) {
        throw std::invalid_argument("R-GCN training node feature shape is invalid");
    }
    if (relation_weights.size() != config.relation_count * config.feature_dim) {
        throw std::invalid_argument("R-GCN training relation weight shape is invalid");
    }

    constexpr float kEps = 0.001f;
    RgcnTrainResult result;
    result.loss_history.reserve(config.epochs);
    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        const float epoch_loss = RgcnReconstructionLoss(
            triples,
            node_features,
            relation_weights,
            config.node_count,
            config.relation_count,
            config.feature_dim);
        result.loss_history.push_back(epoch_loss);

        for (float& value : relation_weights) {
            const float original = value;
            value = original + kEps;
            const float plus_loss = RgcnReconstructionLoss(
                triples,
                node_features,
                relation_weights,
                config.node_count,
                config.relation_count,
                config.feature_dim);
            value = original - kEps;
            const float minus_loss = RgcnReconstructionLoss(
                triples,
                node_features,
                relation_weights,
                config.node_count,
                config.relation_count,
                config.feature_dim);
            value = original;
            const float grad = (plus_loss - minus_loss) / (2.0f * kEps);
            value -= config.learning_rate * grad;
        }
    }
    return result;
}

void SaveRgcnRelationCheckpoint(
    const std::string& path,
    std::size_t node_count,
    std::size_t relation_count,
    std::size_t feature_dim,
    const std::vector<float>& relation_weights)
{
    if (node_count == 0 || relation_count == 0 || feature_dim == 0) {
        throw std::invalid_argument("R-GCN checkpoint dimensions must be greater than zero");
    }
    if (relation_weights.size() != relation_count * feature_dim) {
        throw std::invalid_argument("R-GCN relation weights do not match checkpoint shape");
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open R-GCN checkpoint for writing: " + path);
    }

    output << kRgcnRelationCheckpointMagic << '\n';
    output << node_count << ' ' << relation_count << ' ' << feature_dim << '\n';
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t i = 0; i < relation_weights.size(); ++i) {
        if (i != 0) {
            output << ' ';
        }
        output << relation_weights[i];
    }
    output << '\n';

    if (!output) {
        throw std::runtime_error("failed to write R-GCN checkpoint: " + path);
    }
}

void LoadRgcnRelationCheckpoint(
    const std::string& path,
    std::size_t expected_node_count,
    std::size_t expected_relation_count,
    std::size_t expected_feature_dim,
    std::vector<float>& relation_weights)
{
    if (expected_node_count == 0 || expected_relation_count == 0 || expected_feature_dim == 0) {
        throw std::invalid_argument("R-GCN checkpoint dimensions must be greater than zero");
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open R-GCN checkpoint for reading: " + path);
    }

    std::string magic;
    input >> magic;
    if (magic != kRgcnRelationCheckpointMagic) {
        throw std::runtime_error("invalid R-GCN checkpoint magic");
    }

    std::size_t node_count = 0;
    std::size_t relation_count = 0;
    std::size_t feature_dim = 0;
    input >> node_count >> relation_count >> feature_dim;
    if (node_count != expected_node_count ||
        relation_count != expected_relation_count ||
        feature_dim != expected_feature_dim) {
        throw std::runtime_error("R-GCN checkpoint shape does not match requested shape");
    }

    relation_weights.assign(relation_count * feature_dim, 0.0f);
    for (float& value : relation_weights) {
        input >> value;
    }
    if (!input) {
        throw std::runtime_error("R-GCN checkpoint ended before all relation weights were read");
    }
}

} // namespace minmind
