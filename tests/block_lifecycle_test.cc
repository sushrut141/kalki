#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "common/fake_clients.h"
#include "gtest/gtest.h"
#include "kalki/common/config.h"
#include "kalki/core/database_engine.h"
#include "kalki/metadata/metadata_store.h"
#include "kalki/storage/wal.h"

namespace {

std::string CreateTempDir() {
  const auto dir = std::filesystem::temp_directory_path() /
                   absl::StrCat("kalki_wal_block_", absl::ToUnixMicros(absl::Now()));
  std::filesystem::create_directories(dir);
  return dir.string();
}

kalki::DatabaseConfig BuildWalBlockConfig(const std::string& base_dir) {
  kalki::DatabaseConfig cfg;
  cfg.data_dir = base_dir;
  cfg.wal_path = base_dir + "/wal.log";
  cfg.metadata_db_path = base_dir + "/metadata.db";
  cfg.fresh_block_dir = base_dir + "/blocks/fresh";
  cfg.baked_block_dir = base_dir + "/blocks/baked";
  cfg.grpc_listen_address = "127.0.0.1:0";
  cfg.max_records_per_fresh_block = 10;
  cfg.wal_trim_threshold_records = 20;
  cfg.wal_read_batch_size = 256;
  cfg.query_worker_threads = 2;
  cfg.similarity_threshold = -1.0;
  cfg.query_timeout_ms = 2000;
  cfg.worker_timeout_ms = 200;
  cfg.ingestion_poll_interval = absl::Milliseconds(5);
  cfg.compaction_poll_interval = absl::Milliseconds(5);
  return cfg;
}

TEST(BlockLifecycleTest, InitializingDatabaseEngineCreatesWalOnDisk) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildWalBlockConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  EXPECT_TRUE(init_status.ok());
  EXPECT_TRUE(std::filesystem::exists(config.data_dir));
  EXPECT_TRUE(std::filesystem::exists(config.metadata_db_path));

  const bool wal_exists = std::filesystem::exists(config.wal_path);
  EXPECT_TRUE(wal_exists);

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

TEST(BlockLifecycleTest, StoringLogsIncrementsWalCount) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildWalBlockConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  auto* metadata_store = engine.GetMetadataStoreForTest();
  EXPECT_TRUE(init_status.ok());
  EXPECT_NE(metadata_store, nullptr);

  for (int i = 0; i < 6; ++i) {
    const auto append_status = engine.AppendConversation(
        "agent_a", absl::StrCat("session_", i), absl::StrCat("conversation_", i), absl::Now());
    EXPECT_TRUE(append_status.ok());
  }

  absl::SleepFor(absl::Seconds(1));

  const auto wal_count_or = metadata_store->GetWalRecordCount();
  EXPECT_TRUE(wal_count_or.ok());
  EXPECT_EQ(*wal_count_or, 6);

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

TEST(BlockLifecycleTest, StoringLogsCreatesFreshBlockOnDisk) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildWalBlockConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  auto* metadata_store = engine.GetMetadataStoreForTest();
  EXPECT_TRUE(init_status.ok());
  EXPECT_NE(metadata_store, nullptr);
  EXPECT_TRUE(std::filesystem::exists(config.fresh_block_dir));
  for (int i = 0; i < 3; ++i) {
    const auto append_status = engine.AppendConversation("agent_fresh", absl::StrCat("session_", i),
                                                         absl::StrCat("fresh_", i), absl::Now());
    EXPECT_TRUE(append_status.ok());
  }

  absl::SleepFor(absl::Seconds(1));

  const auto active_fresh_or = metadata_store->GetActiveFreshBlock();
  EXPECT_TRUE(active_fresh_or.ok());
  EXPECT_TRUE(active_fresh_or->has_value());
  EXPECT_TRUE(std::filesystem::exists((*active_fresh_or)->block_path));
  EXPECT_EQ(std::filesystem::path((*active_fresh_or)->block_path).extension(), ".fblk");

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

TEST(BlockLifecycleTest, StoringLogsEventuallyLeadsToCreationOfABakedBlock) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildWalBlockConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  auto* metadata_store = engine.GetMetadataStoreForTest();

  EXPECT_TRUE(init_status.ok());
  EXPECT_NE(metadata_store, nullptr);
  EXPECT_TRUE(std::filesystem::exists(config.baked_block_dir));

  for (int i = 0; i < 14; ++i) {
    const auto append_status = engine.AppendConversation("agent_baked", absl::StrCat("session_", i),
                                                         absl::StrCat("baked_", i), absl::Now());
    EXPECT_TRUE(append_status.ok());
  }

  absl::SleepFor(absl::Seconds(1));

  const kalki::QueryFilter filter;
  const auto baked_blocks_or = metadata_store->FindCandidateBakedBlocks(filter);
  EXPECT_TRUE(baked_blocks_or.ok());
  EXPECT_EQ(baked_blocks_or->size(), 1);
  EXPECT_TRUE(std::filesystem::exists((*baked_blocks_or)[0].block_path));
  EXPECT_EQ(std::filesystem::path((*baked_blocks_or)[0].block_path).extension(), ".bblk");
  EXPECT_EQ((*baked_blocks_or)[0].record_count, config.max_records_per_fresh_block);

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

TEST(BlockLifecycleTest, PostCreationOfABakedBlockAFreshBlockIsAlsoCreated) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildWalBlockConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  auto* metadata_store = engine.GetMetadataStoreForTest();

  EXPECT_TRUE(init_status.ok());
  EXPECT_NE(metadata_store, nullptr);
  EXPECT_TRUE(std::filesystem::exists(config.baked_block_dir));
  EXPECT_TRUE(std::filesystem::exists(config.fresh_block_dir));

  for (int i = 0; i < 16; ++i) {
    const auto append_status = engine.AppendConversation("agent_mix", absl::StrCat("session_", i),
                                                         absl::StrCat("mix_", i), absl::Now());
    EXPECT_TRUE(append_status.ok());
  }

  absl::SleepFor(absl::Seconds(1));

  const kalki::QueryFilter filter;
  const auto baked_blocks_or = metadata_store->FindCandidateBakedBlocks(filter);
  const auto active_fresh_or = metadata_store->GetActiveFreshBlock();

  EXPECT_TRUE(baked_blocks_or.ok());
  EXPECT_FALSE(baked_blocks_or->empty());
  EXPECT_TRUE(std::filesystem::exists((*baked_blocks_or)[0].block_path));
  EXPECT_EQ(std::filesystem::path((*baked_blocks_or)[0].block_path).extension(), ".bblk");
  EXPECT_TRUE(active_fresh_or.ok());
  EXPECT_TRUE(active_fresh_or->has_value());
  EXPECT_TRUE(std::filesystem::exists((*active_fresh_or)->block_path));
  EXPECT_EQ(std::filesystem::path((*active_fresh_or)->block_path).extension(), ".fblk");

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

TEST(BlockLifecycleTest, ExceedingWalTrimThresholdCausesWalTrimming) {
  const std::string base_dir = CreateTempDir();
  kalki::DatabaseConfig config = BuildWalBlockConfig(base_dir);
  config.max_records_per_fresh_block = 10;
  config.wal_trim_threshold_records = 12;
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  auto* metadata_store = engine.GetMetadataStoreForTest();
  EXPECT_TRUE(init_status.ok());
  EXPECT_NE(metadata_store, nullptr);

  for (int i = 0; i < 30; ++i) {
    const auto append_status = engine.AppendConversation("agent_trim", absl::StrCat("session_", i),
                                                         absl::StrCat("trim_", i), absl::Now());
    EXPECT_TRUE(append_status.ok());
  }
  absl::SleepFor(absl::Seconds(1));

  const auto wal_count_or = metadata_store->GetWalRecordCount();
  const int64_t final_count = wal_count_or.ok() ? *wal_count_or : -1;
  EXPECT_TRUE(wal_count_or.ok());
  EXPECT_LE(final_count, config.wal_trim_threshold_records);
  EXPECT_LT(final_count, 30);

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

TEST(BlockLifecycleTest, StoringLogsShowsCorrectCountOfRecordsAfterWalTrimming) {
  const std::string base_dir = CreateTempDir();
  kalki::DatabaseConfig config = BuildWalBlockConfig(base_dir);
  config.max_records_per_fresh_block = 10;
  config.wal_trim_threshold_records = 12;
  const int appended_records = 27;
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  auto* metadata_store = engine.GetMetadataStoreForTest();
  auto* wal_store = engine.GetWalStoreForTest();
  EXPECT_TRUE(init_status.ok());
  EXPECT_NE(metadata_store, nullptr);
  EXPECT_NE(wal_store, nullptr);
  EXPECT_TRUE(std::filesystem::exists(config.wal_path));
  EXPECT_GT(config.wal_trim_threshold_records, config.max_records_per_fresh_block);
  EXPECT_GT(appended_records, config.wal_trim_threshold_records);
  for (int i = 0; i < appended_records; ++i) {
    const auto append_status =
        engine.AppendConversation("agent_trim_count", absl::StrCat("session_", i),
                                  absl::StrCat("trim_count_", i), absl::Now());
    EXPECT_TRUE(append_status.ok());
  }
  absl::SleepFor(absl::Seconds(1));

  const auto wal_count_or = metadata_store->GetWalRecordCount();
  const int64_t final_count = wal_count_or.ok() ? *wal_count_or : -1;
  const auto wal_records_or = wal_store->ReadBatchFromOffset(0, appended_records);
  EXPECT_TRUE(wal_records_or.ok());
  const int64_t wal_records_on_disk =
      wal_records_or.ok() ? static_cast<int64_t>(wal_records_or->size()) : -1;

  EXPECT_TRUE(wal_count_or.ok());
  EXPECT_LE(final_count, config.wal_trim_threshold_records);
  EXPECT_LT(final_count, appended_records);
  EXPECT_EQ(wal_records_on_disk, final_count);
  EXPECT_LT(wal_records_on_disk, appended_records);

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

}  // namespace
