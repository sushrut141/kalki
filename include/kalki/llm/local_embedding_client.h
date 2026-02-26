#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "kalki/llm/embedding_client.h"
#include "llama.h"

namespace kalki {

class LocalEmbeddingClient final : public EmbeddingClient {
 public:
  LocalEmbeddingClient(std::string model_path, int thread_count);
  ~LocalEmbeddingClient() override;
  LocalEmbeddingClient(const LocalEmbeddingClient&) = delete;
  LocalEmbeddingClient& operator=(const LocalEmbeddingClient&) = delete;
  absl::Status Initialize() override;

  absl::StatusOr<std::vector<float>> EmbedText(const std::string& text) override;

 private:
  absl::Status InitializeLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::StatusOr<std::vector<int32_t>> TokenizeLocked(const std::string& text)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  static void Normalize(std::vector<float>* vector_values);

  const std::string model_path_;
  const int thread_count_;
  absl::Mutex mu_;
  llama_model* model_ ABSL_GUARDED_BY(mu_) = nullptr;
  llama_context* context_ ABSL_GUARDED_BY(mu_) = nullptr;
  const llama_vocab* vocab_ ABSL_GUARDED_BY(mu_) = nullptr;
  int32_t embedding_dims_ ABSL_GUARDED_BY(mu_) = 0;
  bool use_encoder_ ABSL_GUARDED_BY(mu_) = false;
  bool initialized_ ABSL_GUARDED_BY(mu_) = false;
};

}  // namespace kalki
