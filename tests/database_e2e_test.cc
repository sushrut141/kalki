#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "kalki/common/config.h"
#include "kalki/common/types.h"
#include "kalki/core/database_engine.h"
#include "kalki/llm/embedding_client.h"
#include "kalki/llm/llm_client.h"

namespace {

class FakeLlmClient final : public kalki::LlmClient {
 public:
  absl::StatusOr<std::vector<std::string>> SummarizeConversation(
      const std::string& conversation_log) override {
    return std::vector<std::string>{absl::StrCat("summary: ", conversation_log)};
  }
};

class FakeEmbeddingClient final : public kalki::EmbeddingClient {
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

std::string CreateTempDir() {
  const auto dir = std::filesystem::temp_directory_path() /
                   absl::StrCat("kalki_e2e_", absl::ToUnixMicros(absl::Now()));
  std::filesystem::create_directories(dir);
  return dir.string();
}

kalki::DatabaseConfig BuildTestConfig(const std::string& base_dir) {
  kalki::DatabaseConfig cfg;
  cfg.data_dir = base_dir;
  cfg.wal_path = base_dir + "/wal.log";
  cfg.metadata_db_path = base_dir + "/metadata.db";
  cfg.fresh_block_dir = base_dir + "/blocks/fresh";
  cfg.baked_block_dir = base_dir + "/blocks/baked";
  cfg.grpc_listen_address = "127.0.0.1:0";
  cfg.llm_api_key = "unused";
  cfg.llm_model = "unused";

  cfg.max_records_per_fresh_block = 1;
  cfg.wal_read_batch_size = 100;
  cfg.query_worker_threads = 2;
  cfg.similarity_threshold = -1.0;
  cfg.query_timeout_ms = 2000;
  cfg.worker_timeout_ms = 200;
  cfg.ingestion_poll_interval = absl::Milliseconds(5);
  cfg.compaction_poll_interval = absl::Milliseconds(5);
  return cfg;
}

std::optional<kalki::QueryExecutionResult> WaitForQueryRecords(kalki::DatabaseEngine* engine,
                                                               size_t expected) {
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (absl::Now() < deadline) {
    kalki::QueryFilter filter;
    filter.caller_agent_id = "test_caller";

    auto result_or = engine->QueryLogs("stored conversations", filter);
    if (result_or.ok() && result_or->status == kalki::QueryCompletionStatus::kComplete &&
        result_or->records.size() >= expected) {
      return result_or.value();
    }
    absl::SleepFor(absl::Milliseconds(25));
  }
  return std::nullopt;
}

class DatabaseE2ETest : public ::testing::Test {
 protected:
  void SetUp() override {
    base_dir_ = CreateTempDir();
    config_ = BuildTestConfig(base_dir_);
  }

  void TearDown() override {
    if (engine_ != nullptr) {
      engine_->Shutdown();
    }
    engine_.reset();

    std::error_code ec;
    std::filesystem::remove_all(base_dir_, ec);
  }

  std::string base_dir_;
  kalki::DatabaseConfig config_;
  std::unique_ptr<kalki::DatabaseEngine> engine_;
};

TEST_F(DatabaseE2ETest, DatabaseIsInitialized) {
  engine_ = std::make_unique<kalki::DatabaseEngine>(config_, std::make_unique<FakeLlmClient>(),
                                                    std::make_unique<FakeEmbeddingClient>());

  auto init_status = engine_->Initialize();
  EXPECT_TRUE(init_status.ok());
  EXPECT_TRUE(std::filesystem::exists(config_.data_dir));
  EXPECT_TRUE(std::filesystem::exists(config_.fresh_block_dir));
  EXPECT_TRUE(std::filesystem::exists(config_.baked_block_dir));
}

TEST_F(DatabaseE2ETest, StoreLogSucceeds) {
  engine_ = std::make_unique<kalki::DatabaseEngine>(config_, std::make_unique<FakeLlmClient>(),
                                                    std::make_unique<FakeEmbeddingClient>());

  auto init_status = engine_->Initialize();
  EXPECT_TRUE(init_status.ok());

  const absl::Time now = absl::Now();
  auto add1 = engine_->AppendConversation("agent_a", "session_1", "conversation one", now);
  auto add2 = engine_->AppendConversation("agent_b", "session_2", "conversation two",
                                          now + absl::Seconds(1));
  auto add3 = engine_->AppendConversation("agent_c", "session_3", "conversation three",
                                          now + absl::Seconds(2));

  EXPECT_TRUE(add1.ok());
  EXPECT_TRUE(add2.ok());
  EXPECT_TRUE(add3.ok());
}

TEST_F(DatabaseE2ETest, QueryLogSucceeds) {
  engine_ = std::make_unique<kalki::DatabaseEngine>(config_, std::make_unique<FakeLlmClient>(),
                                                    std::make_unique<FakeEmbeddingClient>());

  auto init_status = engine_->Initialize();
  EXPECT_TRUE(init_status.ok());

  const absl::Time now = absl::Now();
  auto add1 = engine_->AppendConversation("agent_a", "session_1", "conversation one", now);
  auto add2 = engine_->AppendConversation("agent_b", "session_2", "conversation two",
                                          now + absl::Seconds(1));
  auto add3 = engine_->AppendConversation("agent_c", "session_3", "conversation three",
                                          now + absl::Seconds(2));

  EXPECT_TRUE(add1.ok());
  EXPECT_TRUE(add2.ok());
  EXPECT_TRUE(add3.ok());

  auto result_opt = WaitForQueryRecords(engine_.get(), 3);
  ASSERT_TRUE(result_opt.has_value()) << "Timed out waiting for query to return expected records";

  const kalki::QueryExecutionResult& result = result_opt.value();
  EXPECT_EQ(result.status, kalki::QueryCompletionStatus::kComplete);
  EXPECT_EQ(result.records.size(), 3u);
}

}  // namespace
