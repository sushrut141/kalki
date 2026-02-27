#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "kalki/llm/embedding_client.h"
#include "kalki/llm/llm_client.h"

namespace kalki::test {

class FakeLlmClient final : public LlmClient {
 public:
  absl::StatusOr<std::vector<std::string>> SummarizeConversation(
      const std::string& conversation_log) override {
    return std::vector<std::string>{absl::StrCat("summary: ", conversation_log)};
  }
};

class FakeEmbeddingClient final : public EmbeddingClient {
 public:
  absl::StatusOr<std::vector<float>> EmbedText(const std::string& text) override {
    std::vector<float> v(8, 0.0f);
    for (char c : text) {
      const size_t idx = static_cast<size_t>(static_cast<unsigned char>(c)) % v.size();
      v[idx] += 1.0f;
    }
    return v;
  }
};

}  // namespace kalki::test
