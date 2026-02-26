#pragma once

#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace kalki {

class LlmClient {
 public:
  virtual ~LlmClient() = default;

  virtual absl::StatusOr<std::vector<std::string>> SummarizeConversation(const std::string& conversation_log) = 0;
};

}  // namespace kalki
