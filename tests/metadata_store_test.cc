#include "kalki/metadata/metadata_store.h"

#include <filesystem>
#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "common/fake_clients.h"
#include "gtest/gtest.h"
#include "kalki/common/config.h"
#include "kalki/core/database_engine.h"

namespace {

std::string CreateTempDir() {
  const auto dir = std::filesystem::temp_directory_path() /
                   absl::StrCat("kalki_metadata_store_", absl::ToUnixMicros(absl::Now()));
  std::filesystem::create_directories(dir);
  return dir.string();
}

kalki::DatabaseConfig BuildMetadataStoreTestConfig(const std::string& base_dir) {
  kalki::DatabaseConfig cfg;
  cfg.data_dir = base_dir;
  cfg.wal_path = base_dir + "/wal.log";
  cfg.metadata_db_path = base_dir + "/metadata.db";
  cfg.fresh_block_dir = base_dir + "/blocks/fresh";
  cfg.baked_block_dir = base_dir + "/blocks/baked";
  cfg.grpc_listen_address = "127.0.0.1:0";
  cfg.llm_api_key = "unused";
  cfg.llm_model = "unused";
  cfg.max_records_per_fresh_block = 10;
  cfg.wal_trim_threshold_records = 200;
  cfg.wal_read_batch_size = 512;
  cfg.query_worker_threads = 2;
  cfg.similarity_threshold = -1.0;
  cfg.query_timeout_ms = 2000;
  cfg.worker_timeout_ms = 200;
  cfg.ingestion_poll_interval = absl::Milliseconds(5);
  cfg.compaction_poll_interval = absl::Milliseconds(5);
  return cfg;
}

TEST(MetadataStoreTest, BlocksAreFilteredByAgentIdInFindCandidateBakedBlocks) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildMetadataStoreTestConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  auto* metadata_store = engine.GetMetadataStoreForTest();
  const absl::Time base_time = absl::Now() - absl::Minutes(20);
  EXPECT_TRUE(init_status.ok());
  EXPECT_NE(metadata_store, nullptr);
  EXPECT_TRUE(std::filesystem::exists(config.baked_block_dir));
  for (int i = 0; i < 240; ++i) {
    const auto append_status =
        engine.AppendConversation(absl::StrCat("agent_", i % 6), absl::StrCat("session_", i % 8),
                                  absl::StrCat("conversation_", i), base_time + absl::Seconds(i));
    EXPECT_TRUE(append_status.ok());
  }
  absl::SleepFor(absl::Seconds(2));

  kalki::QueryFilter all_filter;
  const auto all_candidates_or = metadata_store->FindCandidateBakedBlocks(all_filter);
  kalki::QueryFilter missing_agent_filter;
  missing_agent_filter.agent_id = "agent_not_present_in_dataset";
  const auto missing_agent_candidates_or =
      metadata_store->FindCandidateBakedBlocks(missing_agent_filter);

  EXPECT_TRUE(all_candidates_or.ok());
  EXPECT_TRUE(missing_agent_candidates_or.ok());
  EXPECT_GT(all_candidates_or->size(), 0u);
  EXPECT_EQ(missing_agent_candidates_or->size(), 0u);

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

TEST(MetadataStoreTest, BlocksAreFilteredBySessionIdInFindCandidateBakedBlocks) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildMetadataStoreTestConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  auto* metadata_store = engine.GetMetadataStoreForTest();
  const absl::Time base_time = absl::Now() - absl::Minutes(20);
  EXPECT_TRUE(init_status.ok());
  EXPECT_NE(metadata_store, nullptr);
  EXPECT_TRUE(std::filesystem::exists(config.baked_block_dir));
  for (int i = 0; i < 240; ++i) {
    const auto append_status =
        engine.AppendConversation(absl::StrCat("agent_", i % 6), absl::StrCat("session_", i % 8),
                                  absl::StrCat("conversation_", i), base_time + absl::Seconds(i));
    EXPECT_TRUE(append_status.ok());
  }
  absl::SleepFor(absl::Seconds(2));

  kalki::QueryFilter all_filter;
  const auto all_candidates_or = metadata_store->FindCandidateBakedBlocks(all_filter);
  kalki::QueryFilter missing_session_filter;
  missing_session_filter.session_id = "session_not_present_in_dataset";
  const auto missing_session_candidates_or =
      metadata_store->FindCandidateBakedBlocks(missing_session_filter);

  EXPECT_TRUE(all_candidates_or.ok());
  EXPECT_TRUE(missing_session_candidates_or.ok());
  EXPECT_GT(all_candidates_or->size(), 0u);
  EXPECT_EQ(missing_session_candidates_or->size(), 0u);

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

}  // namespace
