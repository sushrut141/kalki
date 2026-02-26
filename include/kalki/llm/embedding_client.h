#pragma once

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace kalki {

class EmbeddingClient {
 public:
  virtual ~EmbeddingClient() = default;
  virtual absl::Status Initialize() { return absl::OkStatus(); }
  virtual absl::StatusOr<std::vector<float>> EmbedText(const std::string& text) = 0;
};

}  // namespace kalki
