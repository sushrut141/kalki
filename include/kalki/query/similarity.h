#pragma once

#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace kalki {

class SimilarityEngine {
 public:
  explicit SimilarityEngine(int dimensions = 128);

  absl::StatusOr<std::vector<float>> ScoreSummaries(
      const std::string& query, const std::vector<std::string>& summaries) const;

 private:
  std::vector<float> Embed(const std::string& text) const;

  int dimensions_;
};

}  // namespace kalki
