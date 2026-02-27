#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "common/fake_clients.h"
#include "gtest/gtest.h"
#include "kalki/common/config.h"
#include "kalki/common/types.h"
#include "kalki/core/database_engine.h"

namespace {

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

TEST(DatabaseE2ETest, DatabaseIsInitialized) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildTestConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());

  const auto init_status = engine.Initialize();

  EXPECT_TRUE(init_status.ok());
  EXPECT_TRUE(std::filesystem::exists(config.data_dir));
  EXPECT_TRUE(std::filesystem::exists(config.fresh_block_dir));
  EXPECT_TRUE(std::filesystem::exists(config.baked_block_dir));

  engine.Shutdown();
  std::error_code ec;
  std::filesystem::remove_all(base_dir, ec);
}

TEST(DatabaseE2ETest, StoreLogSucceeds) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildTestConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  const absl::Time now = absl::Now();

  const auto add1 = engine.AppendConversation("agent_a", "session_1", "conversation one", now);
  const auto add2 =
      engine.AppendConversation("agent_b", "session_2", "conversation two", now + absl::Seconds(1));
  const auto add3 = engine.AppendConversation("agent_c", "session_3", "conversation three",
                                              now + absl::Seconds(2));

  EXPECT_TRUE(init_status.ok());
  EXPECT_TRUE(add1.ok());
  EXPECT_TRUE(add2.ok());
  EXPECT_TRUE(add3.ok());

  engine.Shutdown();
  std::error_code ec;
  std::filesystem::remove_all(base_dir, ec);
}

TEST(DatabaseE2ETest, QueryLogSucceeds) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildTestConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  EXPECT_TRUE(init_status.ok());
  const absl::Time now = absl::Now();
  const auto add1 = engine.AppendConversation("agent_a", "session_1", "conversation one", now);
  const auto add2 =
      engine.AppendConversation("agent_b", "session_2", "conversation two", now + absl::Seconds(1));
  const auto add3 = engine.AppendConversation("agent_c", "session_3", "conversation three",
                                              now + absl::Seconds(2));
  EXPECT_TRUE(add1.ok());
  EXPECT_TRUE(add2.ok());
  EXPECT_TRUE(add3.ok());
  kalki::QueryExecutionResult query_result;
  bool found = false;

  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (absl::Now() < deadline) {
    kalki::QueryFilter filter;
    filter.caller_agent_id = "test_caller";
    const auto result_or = engine.QueryLogs("stored conversations", filter);
    if (result_or.ok() && result_or->status == kalki::QueryCompletionStatus::kComplete &&
        result_or->records.size() == 3) {
      query_result = *result_or;
      found = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(25));
  }

  EXPECT_TRUE(found);
  EXPECT_EQ(query_result.status, kalki::QueryCompletionStatus::kComplete);
  EXPECT_EQ(query_result.records.size(), 3u);

  engine.Shutdown();
  std::error_code ec;
  std::filesystem::remove_all(base_dir, ec);
}

TEST(DatabaseE2ETest, FilterByAgentID) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildTestConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  EXPECT_TRUE(init_status.ok());
  const absl::Time now = absl::Now();
  const auto add1 = engine.AppendConversation("agent_a", "session_1", "conversation one", now);
  const auto add2 =
      engine.AppendConversation("agent_b", "session_2", "conversation two", now + absl::Seconds(1));
  const auto add3 = engine.AppendConversation("agent_c", "session_3", "conversation three",
                                              now + absl::Seconds(2));
  EXPECT_TRUE(add1.ok());
  EXPECT_TRUE(add2.ok());
  EXPECT_TRUE(add3.ok());
  kalki::QueryExecutionResult query_result;
  bool found = false;

  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (absl::Now() < deadline) {
    kalki::QueryFilter filter;
    filter.caller_agent_id = "test_caller";
    filter.agent_id = "agent_b";
    const auto result_or = engine.QueryLogs("stored conversations", filter);
    if (result_or.ok() && result_or->status == kalki::QueryCompletionStatus::kComplete &&
        result_or->records.size() == 1) {
      query_result = *result_or;
      found = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(25));
  }

  EXPECT_TRUE(found);
  EXPECT_EQ(query_result.records.size(), 1u);
  EXPECT_EQ(query_result.records[0].agent_id, "agent_b");

  engine.Shutdown();
  std::error_code ec;
  std::filesystem::remove_all(base_dir, ec);
}

TEST(DatabaseE2ETest, FilterBySessionID) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildTestConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  EXPECT_TRUE(init_status.ok());
  const absl::Time now = absl::Now();
  const auto add1 = engine.AppendConversation("agent_a", "session_1", "conversation one", now);
  const auto add2 =
      engine.AppendConversation("agent_b", "session_2", "conversation two", now + absl::Seconds(1));
  const auto add3 = engine.AppendConversation("agent_c", "session_3", "conversation three",
                                              now + absl::Seconds(2));
  EXPECT_TRUE(add1.ok());
  EXPECT_TRUE(add2.ok());
  EXPECT_TRUE(add3.ok());
  kalki::QueryExecutionResult query_result;
  bool found = false;

  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (absl::Now() < deadline) {
    kalki::QueryFilter filter;
    filter.caller_agent_id = "test_caller";
    filter.session_id = "session_3";
    const auto result_or = engine.QueryLogs("stored conversations", filter);
    if (result_or.ok() && result_or->status == kalki::QueryCompletionStatus::kComplete &&
        result_or->records.size() == 1) {
      query_result = *result_or;
      found = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(25));
  }

  EXPECT_TRUE(found);
  EXPECT_EQ(query_result.records.size(), 1u);
  EXPECT_EQ(query_result.records[0].session_id, "session_3");

  engine.Shutdown();
  std::error_code ec;
  std::filesystem::remove_all(base_dir, ec);
}

TEST(DatabaseE2ETest, FilterByTimestamp) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildTestConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  EXPECT_TRUE(init_status.ok());
  const absl::Time now = absl::Now();
  const auto ts1 = now;
  const auto ts2 = now + absl::Seconds(1);
  const auto ts3 = now + absl::Seconds(2);
  const auto add1 = engine.AppendConversation("agent_a", "session_1", "conversation one", ts1);
  const auto add2 = engine.AppendConversation("agent_b", "session_2", "conversation two", ts2);
  const auto add3 = engine.AppendConversation("agent_c", "session_3", "conversation three", ts3);
  EXPECT_TRUE(add1.ok());
  EXPECT_TRUE(add2.ok());
  EXPECT_TRUE(add3.ok());
  kalki::QueryExecutionResult query_result;
  bool found = false;

  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (absl::Now() < deadline) {
    kalki::QueryFilter filter;
    filter.caller_agent_id = "test_caller";
    filter.start_time = ts2;
    filter.end_time = ts2;
    const auto result_or = engine.QueryLogs("stored conversations", filter);
    if (result_or.ok() && result_or->status == kalki::QueryCompletionStatus::kComplete &&
        result_or->records.size() == 1) {
      query_result = *result_or;
      found = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(25));
  }

  EXPECT_TRUE(found);
  EXPECT_EQ(query_result.records.size(), 1u);
  EXPECT_EQ(query_result.records[0].agent_id, "agent_b");
  EXPECT_EQ(query_result.records[0].session_id, "session_2");

  engine.Shutdown();
  std::error_code ec;
  std::filesystem::remove_all(base_dir, ec);
}

TEST(DatabaseE2ETest, AllFilters) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildTestConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  EXPECT_TRUE(init_status.ok());
  const absl::Time now = absl::Now();
  const auto ts1 = now;
  const auto ts2 = now + absl::Seconds(1);
  const auto ts3 = now + absl::Seconds(2);
  const auto add1 = engine.AppendConversation("agent_a", "session_1", "conversation one", ts1);
  const auto add2 = engine.AppendConversation("agent_b", "session_2", "conversation two", ts2);
  const auto add3 = engine.AppendConversation("agent_c", "session_3", "conversation three", ts3);
  EXPECT_TRUE(add1.ok());
  EXPECT_TRUE(add2.ok());
  EXPECT_TRUE(add3.ok());
  kalki::QueryExecutionResult query_result;
  bool found = false;

  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (absl::Now() < deadline) {
    kalki::QueryFilter filter;
    filter.caller_agent_id = "test_caller";
    filter.agent_id = "agent_c";
    filter.session_id = "session_3";
    filter.start_time = ts3;
    filter.end_time = ts3;
    const auto result_or = engine.QueryLogs("stored conversations", filter);
    if (result_or.ok() && result_or->status == kalki::QueryCompletionStatus::kComplete &&
        result_or->records.size() == 1) {
      query_result = *result_or;
      found = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(25));
  }

  EXPECT_TRUE(found);
  EXPECT_EQ(query_result.records.size(), 1u);
  EXPECT_EQ(query_result.records[0].agent_id, "agent_c");
  EXPECT_EQ(query_result.records[0].session_id, "session_3");

  engine.Shutdown();
  std::error_code ec;
  std::filesystem::remove_all(base_dir, ec);
}

}  // namespace
