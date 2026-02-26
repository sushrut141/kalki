#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "kalki/llm/llm_client.h"

namespace kalki {

class GeminiLlmClient final : public LlmClient {
 public:
  GeminiLlmClient(std::string api_key, std::string model);

  absl::StatusOr<std::vector<std::string>> SummarizeConversation(
      const std::string& conversation_log) override;

 private:
  std::string BuildPrompt(const std::string& conversation_log) const;
  absl::StatusOr<std::string> BuildRequestJson(const std::string& prompt) const;
  absl::StatusOr<std::string> PostGenerateContent(const std::string& request_json) const;
  absl::StatusOr<std::string> ExtractCandidateText(const std::string& response_json) const;
  absl::StatusOr<std::vector<std::string>> ParseSummaryPayload(const std::string& model_text) const;

  std::string api_key_;
  std::string model_;
};

}  // namespace kalki
