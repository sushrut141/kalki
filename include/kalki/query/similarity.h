#pragma once

#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace kalki {

class SimilarityEngine {
 public:
  SimilarityEngine() = default;

  absl::StatusOr<std::vector<float>> ScoreEmbeddings(
      const std::vector<float>& query_embedding,
      const std::vector<std::vector<float>>& summary_embeddings) const;
};

}  // namespace kalki
