#pragma once

#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace kalki {

class SimilarityEngine {
 public:
  struct EmbeddingView {
    const float* data = nullptr;
    size_t dims = 0;
  };

  SimilarityEngine() = default;

  absl::StatusOr<std::vector<float>> ScoreEmbeddings(
      const std::vector<float>& query_embedding,
      const std::vector<EmbeddingView>& summary_embeddings) const;
};

}  // namespace kalki
