#include "kalki/query/similarity.h"

#include <faiss/IndexFlat.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"

namespace kalki {

SimilarityEngine::SimilarityEngine(int dimensions) : dimensions_(dimensions) {}

std::vector<float> SimilarityEngine::Embed(const std::string& text) const {
  std::vector<float> v(static_cast<size_t>(dimensions_), 0.0f);
  auto tokens = absl::StrSplit(text, ' ', absl::SkipWhitespace());
  for (absl::string_view token : tokens) {
    std::string lowered(token);
    absl::AsciiStrToLower(&lowered);
    const size_t idx = absl::HashOf(lowered) % static_cast<size_t>(dimensions_);
    v[idx] += 1.0f;
  }

  float norm = 0.0f;
  norm =
      std::accumulate(v.begin(), v.end(), 0.0f, [](float sum, float x) { return sum + (x * x); });
  norm = std::sqrt(norm);
  if (norm > 0.0f) {
    std::transform(v.begin(), v.end(), v.begin(), [norm](float x) { return x / norm; });
  }
  return v;
}

absl::StatusOr<std::vector<float>> SimilarityEngine::ScoreSummaries(
    const std::string& query, const std::vector<std::string>& summaries) const {
  if (summaries.empty()) {
    return std::vector<float>{};
  }

  const std::vector<float> query_vec = Embed(query);

  std::vector<float> summary_matrix;
  summary_matrix.reserve(summaries.size() * static_cast<size_t>(dimensions_));
  for (const auto& summary : summaries) {
    std::vector<float> vec = Embed(summary);
    summary_matrix.insert(summary_matrix.end(), vec.begin(), vec.end());
  }

  faiss::IndexFlatIP index(dimensions_);
  index.add(static_cast<faiss::idx_t>(summaries.size()), summary_matrix.data());

  const int k = static_cast<int>(summaries.size());
  std::vector<faiss::idx_t> labels(static_cast<size_t>(k), -1);
  std::vector<float> distances(static_cast<size_t>(k), -std::numeric_limits<float>::infinity());
  index.search(1, query_vec.data(), k, distances.data(), labels.data());

  std::vector<float> scores(summaries.size(), -std::numeric_limits<float>::infinity());
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] >= 0 && labels[i] < static_cast<faiss::idx_t>(scores.size())) {
      scores[static_cast<size_t>(labels[i])] = distances[i];
    }
  }
  return scores;
}

}  // namespace kalki
