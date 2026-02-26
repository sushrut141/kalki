#include "kalki/llm/local_embedding_client.h"

#include <filesystem>
#include <string>

#include "gtest/gtest.h"

namespace {

std::string ModelPath() {
  return std::string(KALKI_PROJECT_SOURCE_DIR) + "/third_party/models/all-minilm-v2.gguf";
}

TEST(LocalEmbeddingClientTest, IsInitialized) {
  ASSERT_TRUE(std::filesystem::exists(ModelPath()));
  kalki::LocalEmbeddingClient client(ModelPath(), 2);

  const auto status = client.Initialize();

  EXPECT_TRUE(status.ok()) << status;
}

TEST(LocalEmbeddingClientTest, CreatesTextEmbedding) {
  ASSERT_TRUE(std::filesystem::exists(ModelPath()));
  kalki::LocalEmbeddingClient client(ModelPath(), 2);

  const auto embedding_or = client.EmbedText("agent action log summary");

  EXPECT_TRUE(embedding_or.ok()) << embedding_or.status();
  ASSERT_TRUE(embedding_or.ok());
  EXPECT_FALSE(embedding_or->empty());
}

TEST(LocalEmbeddingClientTest, TextEmbeddingDimensions) {
  ASSERT_TRUE(std::filesystem::exists(ModelPath()));
  kalki::LocalEmbeddingClient client(ModelPath(), 2);

  const auto embedding_or = client.EmbedText("find prior session context");
  EXPECT_TRUE(embedding_or.ok()) << embedding_or.status();
  ASSERT_TRUE(embedding_or.ok());

  EXPECT_EQ(embedding_or->size(), 384);
}

}  // namespace
