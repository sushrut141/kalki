#include "kalki/llm/local_embedding_client.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ggml-backend.h"
#include "llama.h"

namespace kalki {

namespace {

absl::once_flag kLlamaBackendInitOnce;

void InitializeLlamaBackend() { llama_backend_init(); }

}  // namespace

LocalEmbeddingClient::LocalEmbeddingClient(std::string model_path, int thread_count)
    : model_path_(std::move(model_path)), thread_count_(std::max(1, thread_count)) {}

LocalEmbeddingClient::~LocalEmbeddingClient() {
  absl::MutexLock lock(mu_);
  if (context_ != nullptr) {
    llama_free(context_);
    context_ = nullptr;
  }
  if (model_ != nullptr) {
    llama_model_free(model_);
    model_ = nullptr;
  }
}

absl::Status LocalEmbeddingClient::Initialize() {
  absl::MutexLock lock(mu_);
  if (initialized_) {
    return absl::OkStatus();
  }
  return InitializeLocked();
}

absl::Status LocalEmbeddingClient::InitializeLocked() {
  if (!std::filesystem::exists(model_path_)) {
    return absl::NotFoundError(absl::StrCat("embedding model file not found: ", model_path_,
                                            " (expected local all-minilm:v2 gguf file)"));
  }

  absl::call_once(kLlamaBackendInitOnce, InitializeLlamaBackend);

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
  model_params.use_mmap = true;
  model_params.use_mlock = false;
  model_params.use_extra_bufts = false;
  model_params.no_host = false;

  ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  ggml_backend_dev_t devices[2] = {cpu_device, nullptr};
  if (cpu_device != nullptr) {
    model_params.devices = devices;
  }

  model_ = llama_model_load_from_file(model_path_.c_str(), model_params);
  if (model_ == nullptr) {
    return absl::InternalError(
        absl::StrCat("failed to load embedding model from file: ", model_path_));
  }

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = 512;
  context_params.n_batch = 512;
  context_params.n_ubatch = 512;
  context_params.n_seq_max = 1;
  context_params.n_threads = thread_count_;
  context_params.n_threads_batch = thread_count_;
  context_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
  context_params.embeddings = true;
  context_params.offload_kqv = false;
  context_params.op_offload = false;
  context_params.no_perf = true;

  context_ = llama_init_from_model(model_, context_params);
  if (context_ == nullptr) {
    llama_model_free(model_);
    model_ = nullptr;
    return absl::InternalError("failed to initialize llama context for embeddings");
  }

  llama_set_embeddings(context_, true);
  llama_set_causal_attn(context_, false);

  vocab_ = llama_model_get_vocab(model_);
  if (vocab_ == nullptr) {
    return absl::InternalError("llama model did not expose a tokenizer vocabulary");
  }

  embedding_dims_ = llama_model_n_embd_out(model_);
  if (embedding_dims_ <= 0) {
    embedding_dims_ = llama_model_n_embd(model_);
  }
  if (embedding_dims_ <= 0) {
    return absl::InternalError("failed to determine embedding dimension from llama model");
  }

  const bool has_encoder = llama_model_has_encoder(model_);
  const bool has_decoder = llama_model_has_decoder(model_);
  use_encoder_ = has_encoder || !has_decoder;
  initialized_ = true;

  LOG(INFO) << "component=embedding event=model_loaded model_id=all-minilm:v2 path=" << model_path_
            << " dims=" << embedding_dims_ << " threads=" << thread_count_
            << " encoder=" << (use_encoder_ ? "true" : "false")
            << " has_encoder=" << (has_encoder ? "true" : "false")
            << " has_decoder=" << (has_decoder ? "true" : "false");
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int32_t>> LocalEmbeddingClient::TokenizeLocked(const std::string& text) {
  if (text.empty()) {
    return absl::InvalidArgumentError("cannot embed empty text");
  }
  if (vocab_ == nullptr) {
    return absl::FailedPreconditionError("embedding client vocabulary is not initialized");
  }

  int32_t token_capacity = std::max<int32_t>(32, static_cast<int32_t>(text.size()) + 8);
  std::vector<int32_t> tokens(static_cast<size_t>(token_capacity));

  int32_t token_count =
      llama_tokenize(vocab_, text.c_str(), static_cast<int32_t>(text.size()),
                     reinterpret_cast<llama_token*>(tokens.data()), token_capacity, true, false);

  if (token_count < 0 && token_count != std::numeric_limits<int32_t>::min()) {
    token_capacity = -token_count;
    tokens.resize(static_cast<size_t>(token_capacity));
    token_count =
        llama_tokenize(vocab_, text.c_str(), static_cast<int32_t>(text.size()),
                       reinterpret_cast<llama_token*>(tokens.data()), token_capacity, true, false);
  }

  if (token_count <= 0) {
    return absl::InvalidArgumentError("failed to tokenize text for embedding");
  }

  tokens.resize(static_cast<size_t>(token_count));
  return tokens;
}

void LocalEmbeddingClient::Normalize(std::vector<float>* vector_values) {
  if (vector_values == nullptr || vector_values->empty()) {
    return;
  }

  float norm_sq = 0.0F;
  for (const float v : *vector_values) {
    norm_sq += v * v;
  }
  if (norm_sq <= std::numeric_limits<float>::epsilon()) {
    return;
  }

  const float inv_norm = 1.0F / std::sqrt(norm_sq);
  for (float& v : *vector_values) {
    v *= inv_norm;
  }
}

absl::StatusOr<std::vector<float>> LocalEmbeddingClient::EmbedText(const std::string& text) {
  absl::MutexLock lock(mu_);
  if (!initialized_) {
    auto st = InitializeLocked();
    if (!st.ok()) {
      return st;
    }
  }

  auto tokens_or = TokenizeLocked(text);
  if (!tokens_or.ok()) {
    return tokens_or.status();
  }

  const std::vector<int32_t>& tokens = *tokens_or;
  llama_batch batch =
      llama_batch_get_one(reinterpret_cast<llama_token*>(const_cast<int32_t*>(tokens.data())),
                          static_cast<int32_t>(tokens.size()));

  const int32_t eval_status =
      use_encoder_ ? llama_encode(context_, batch) : llama_decode(context_, batch);
  if (eval_status != 0) {
    return absl::InternalError(
        absl::StrCat("llama embedding evaluation failed with status=", eval_status));
  }
  llama_synchronize(context_);

  float* embedding = llama_get_embeddings_seq(context_, 0);
  if (embedding == nullptr) {
    embedding = llama_get_embeddings_ith(context_, -1);
  }
  if (embedding == nullptr) {
    return absl::InternalError("llama did not return embedding output");
  }

  std::vector<float> out(static_cast<size_t>(embedding_dims_));
  std::memcpy(out.data(), embedding, sizeof(float) * out.size());
  Normalize(&out);

  DLOG(INFO) << "component=embedding event=embedded model_id=all-minilm:v2 dims=" << out.size()
             << " tokens=" << tokens.size();
  return out;
}

}  // namespace kalki
