#include "kalki/query/similarity.h"

#include <faiss/IndexFlat.h>

#include <limits>
#include <vector>

#include "absl/status/status.h"

namespace kalki {

absl::StatusOr<std::vector<float>> SimilarityEngine::ScoreEmbeddings(
    const std::vector<float>& query_embedding,
    const std::vector<std::vector<float>>& summary_embeddings) const {
  if (query_embedding.empty()) {
    return absl::InvalidArgumentError("query embedding is empty");
  }
  if (summary_embeddings.empty()) {
    return std::vector<float>{};
  }

  const int dimensions = static_cast<int>(query_embedding.size());
  std::vector<float> summary_matrix;
  summary_matrix.reserve(summary_embeddings.size() * query_embedding.size());
  for (const auto& embedding : summary_embeddings) {
    if (embedding.size() != query_embedding.size()) {
      return absl::InvalidArgumentError(
          "summary embedding dimensions do not match query embedding dimensions");
    }
    summary_matrix.insert(summary_matrix.end(), embedding.begin(), embedding.end());
  }

  faiss::IndexFlatIP index(dimensions);
  index.add(static_cast<faiss::idx_t>(summary_embeddings.size()), summary_matrix.data());

  const int k = static_cast<int>(summary_embeddings.size());
  std::vector<faiss::idx_t> labels(static_cast<size_t>(k), -1);
  std::vector<float> distances(static_cast<size_t>(k), -std::numeric_limits<float>::infinity());
  index.search(1, query_embedding.data(), k, distances.data(), labels.data());

  std::vector<float> scores(summary_embeddings.size(), -std::numeric_limits<float>::infinity());
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] >= 0 && labels[i] < static_cast<faiss::idx_t>(scores.size())) {
      scores[static_cast<size_t>(labels[i])] = distances[i];
    }
  }
  return scores;
}

}  // namespace kalki
