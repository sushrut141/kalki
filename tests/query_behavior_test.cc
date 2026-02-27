#include <algorithm>
#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "common/fake_clients.h"
#include "gtest/gtest.h"
#include "kalki/common/config.h"
#include "kalki/core/database_engine.h"
#include "kalki/metadata/metadata_store.h"

namespace {

constexpr char kRandomConversation[] = "random agent conversation";
constexpr char kPassingConversation[] = "build passes after updating tests";
constexpr char kQueryString[] = "Did the build pass?";

struct QuerySeedRecord {
  std::string agent_id;
  std::string session_id;
  std::string conversation_log;
  absl::Time timestamp;
};

std::string CreateTempDir() {
  const auto dir = std::filesystem::temp_directory_path() /
                   absl::StrCat("kalki_query_behavior_", absl::ToUnixMicros(absl::Now()));
  std::filesystem::create_directories(dir);
  return dir.string();
}

kalki::DatabaseConfig BuildQueryConfig(const std::string& base_dir) {
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
  cfg.wal_trim_threshold_records = 20;
  cfg.wal_read_batch_size = 512;
  cfg.query_worker_threads = 4;
  cfg.similarity_threshold = 100.0;
  cfg.query_timeout_ms = 5000;
  cfg.worker_timeout_ms = 500;
  cfg.ingestion_poll_interval = absl::Milliseconds(5);
  cfg.compaction_poll_interval = absl::Milliseconds(5);
  return cfg;
}

std::vector<QuerySeedRecord> BuildShuffledRecords(
    int total_records, int passing_records, const std::function<std::string(int)>& agent_selector,
    const std::function<std::string(int)>& session_selector,
    const std::function<absl::Time(int)>& timestamp_selector) {
  std::vector<QuerySeedRecord> records;
  records.reserve(static_cast<size_t>(total_records));

  for (int i = 0; i < total_records; ++i) {
    QuerySeedRecord record;
    record.agent_id = agent_selector(i);
    record.session_id = session_selector(i);
    record.conversation_log = i < passing_records ? kPassingConversation : kRandomConversation;
    record.timestamp = timestamp_selector(i);
    records.push_back(std::move(record));
  }

  std::mt19937 rng(static_cast<uint32_t>(total_records * 97 + passing_records * 13 + 11));
  std::shuffle(records.begin(), records.end(), rng);
  return records;
}

TEST(QueryBehaviorTest, QueryWorksWithSmallNumberOfRecords) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildQueryConfig(base_dir);
  const int expected_passing_records = 3;
  const absl::Time base_time = absl::Now() - absl::Minutes(5);
  const std::vector<QuerySeedRecord> records = BuildShuffledRecords(
      18, expected_passing_records, [](int i) { return absl::StrCat("agent_", i % 3); },
      [](int i) { return absl::StrCat("session_", i % 4); },
      [&](int i) { return base_time + absl::Seconds(i); });
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();

  EXPECT_TRUE(init_status.ok());
  EXPECT_EQ(records.size(), 18);
  EXPECT_TRUE(std::filesystem::exists(config.wal_path));

  for (const auto& record : records) {
    const auto append_status = engine.AppendConversation(record.agent_id, record.session_id,
                                                         record.conversation_log, record.timestamp);
    EXPECT_TRUE(append_status.ok());
  }
  absl::SleepFor(absl::Seconds(1));

  kalki::QueryFilter filter;
  filter.caller_agent_id = "query_tester";
  const auto query_or = engine.QueryLogs(kQueryString, filter);

  std::set<std::string> payloads;
  if (query_or.ok()) {
    for (const auto& record : query_or->records) {
      payloads.insert(record.raw_conversation_log);
    }
  }
  EXPECT_TRUE(query_or.ok());
  EXPECT_EQ(static_cast<int>(query_or->records.size()), expected_passing_records);
  EXPECT_EQ(payloads.size(), 1u);
  EXPECT_EQ(*payloads.begin(), kPassingConversation);

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

TEST(QueryBehaviorTest, QueryWorksWithLargeNumberOfRecords) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildQueryConfig(base_dir);
  const int expected_passing_records = 12;
  const absl::Time base_time = absl::Now() - absl::Minutes(30);
  const std::vector<QuerySeedRecord> records = BuildShuffledRecords(
      160, expected_passing_records, [](int i) { return absl::StrCat("agent_", i % 7); },
      [](int i) { return absl::StrCat("session_", i % 9); },
      [&](int i) { return base_time + absl::Seconds(i); });
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  auto* metadata_store = engine.GetMetadataStoreForTest();

  EXPECT_TRUE(init_status.ok());
  EXPECT_NE(metadata_store, nullptr);
  EXPECT_EQ(records.size(), 160);
  EXPECT_TRUE(std::filesystem::exists(config.baked_block_dir));

  for (const auto& record : records) {
    const auto append_status = engine.AppendConversation(record.agent_id, record.session_id,
                                                         record.conversation_log, record.timestamp);
    EXPECT_TRUE(append_status.ok());
  }
  absl::SleepFor(absl::Seconds(1));

  kalki::QueryFilter filter;
  filter.caller_agent_id = "query_tester";
  const auto baked_candidates_or = metadata_store->FindCandidateBakedBlocks(filter);
  const auto query_or = engine.QueryLogs(kQueryString, filter);

  std::set<std::string> payloads;
  if (query_or.ok()) {
    for (const auto& record : query_or->records) {
      payloads.insert(record.raw_conversation_log);
    }
  }
  EXPECT_TRUE(baked_candidates_or.ok());
  EXPECT_GE(static_cast<int>(baked_candidates_or->size()), 2);
  EXPECT_TRUE(query_or.ok());
  EXPECT_EQ(static_cast<int>(query_or->records.size()), expected_passing_records);
  EXPECT_EQ(payloads.size(), 1u);
  EXPECT_EQ(*payloads.begin(), kPassingConversation);

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

TEST(QueryBehaviorTest, QueryWorksInParallelWithIngestion) {
  const std::string base_dir = CreateTempDir();
  kalki::DatabaseConfig config = BuildQueryConfig(base_dir);
  config.max_records_per_fresh_block = 1000;
  const int expected_passing_records = 1;
  const absl::Time base_time = absl::Now() - absl::Minutes(10);
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();

  EXPECT_TRUE(init_status.ok());
  EXPECT_TRUE(std::filesystem::exists(config.fresh_block_dir));
  const auto seed_status =
      engine.AppendConversation("seed_agent", "seed_session", kPassingConversation, base_time);
  EXPECT_TRUE(seed_status.ok());

  std::atomic<bool> append_failed(false);
  std::atomic<bool> query_failed(false);

  std::thread ingest_thread([&engine, &append_failed, base_time]() {
    for (int i = 0; i < 80; ++i) {
      const auto append_status = engine.AppendConversation(
          absl::StrCat("ingest_agent_", i % 5), absl::StrCat("ingest_session_", i % 9),
          kRandomConversation, base_time + absl::Seconds(1 + i));
      if (!append_status.ok()) {
        append_failed.store(true, std::memory_order_relaxed);
      }
      absl::SleepFor(absl::Milliseconds(10));
    }
  });

  std::thread query_thread([&engine, &query_failed, expected_passing_records]() {
    for (int i = 0; i < 10; ++i) {
      kalki::QueryFilter filter;
      filter.caller_agent_id = "query_tester";
      const auto query_or = engine.QueryLogs(kQueryString, filter);
      std::set<std::string> payloads;
      if (!query_or.ok()) {
        query_failed.store(true, std::memory_order_relaxed);
        continue;
      }
      for (const auto& record : query_or->records) {
        payloads.insert(record.raw_conversation_log);
      }
      if (static_cast<int>(query_or->records.size()) != expected_passing_records) {
        query_failed.store(true, std::memory_order_relaxed);
      }
      if (payloads.size() != 1u) {
        query_failed.store(true, std::memory_order_relaxed);
      } else if (*payloads.begin() != kPassingConversation) {
        query_failed.store(true, std::memory_order_relaxed);
      }
      absl::SleepFor(absl::Milliseconds(15));
    }
  });
  ingest_thread.join();
  query_thread.join();

  EXPECT_FALSE(append_failed.load(std::memory_order_relaxed));
  EXPECT_FALSE(query_failed.load(std::memory_order_relaxed));

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

TEST(QueryBehaviorTest, QueryReadsLimitedSetOfBlocksBasedOnAgentIdFilter) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildQueryConfig(base_dir);
  const std::string target_agent = "target_agent";
  const int expected_passing_records = 4;
  const absl::Time base_time = absl::Now() - absl::Minutes(20);
  const std::vector<QuerySeedRecord> records = BuildShuffledRecords(
      120, expected_passing_records,
      [&](int i) {
        return i < expected_passing_records ? target_agent
                                            : absl::StrCat("other_agent_", 1 + (i % 6));
      },
      [](int i) { return absl::StrCat("session_", i % 10); },
      [&](int i) { return base_time + absl::Seconds(i); });
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  auto* metadata_store = engine.GetMetadataStoreForTest();

  EXPECT_TRUE(init_status.ok());
  EXPECT_NE(metadata_store, nullptr);
  EXPECT_EQ(records.size(), 120);
  EXPECT_TRUE(std::filesystem::exists(config.baked_block_dir));

  for (const auto& record : records) {
    const auto append_status = engine.AppendConversation(record.agent_id, record.session_id,
                                                         record.conversation_log, record.timestamp);
    EXPECT_TRUE(append_status.ok());
  }
  absl::SleepFor(absl::Seconds(1));

  kalki::QueryFilter all_filter;
  all_filter.caller_agent_id = "query_tester";
  const auto all_baked_or = metadata_store->FindCandidateBakedBlocks(all_filter);
  const auto all_fresh_or = metadata_store->FindCandidateFreshBlocks(all_filter);
  kalki::QueryFilter filtered;
  filtered.caller_agent_id = "query_tester";
  filtered.agent_id = target_agent;
  const auto filtered_baked_or = metadata_store->FindCandidateBakedBlocks(filtered);
  const auto filtered_fresh_or = metadata_store->FindCandidateFreshBlocks(filtered);

  const int all_block_count = all_baked_or.ok() && all_fresh_or.ok()
                                  ? static_cast<int>(all_baked_or->size() + all_fresh_or->size())
                                  : 0;
  const int filtered_block_count =
      filtered_baked_or.ok() && filtered_fresh_or.ok()
          ? static_cast<int>(filtered_baked_or->size() + filtered_fresh_or->size())
          : 0;

  EXPECT_TRUE(all_baked_or.ok());
  EXPECT_TRUE(all_fresh_or.ok());
  EXPECT_TRUE(filtered_baked_or.ok());
  EXPECT_TRUE(filtered_fresh_or.ok());
  EXPECT_GT(all_block_count, 0);
  EXPECT_GT(filtered_block_count, 0);
  EXPECT_LT(filtered_block_count, all_block_count);

  const auto query_or = engine.QueryLogs(kQueryString, filtered);

  std::set<std::string> payloads;
  if (query_or.ok()) {
    for (const auto& record : query_or->records) {
      payloads.insert(record.raw_conversation_log);
    }
  }
  EXPECT_TRUE(query_or.ok());
  EXPECT_EQ(static_cast<int>(query_or->records.size()), expected_passing_records);
  EXPECT_EQ(payloads.size(), 1u);
  EXPECT_EQ(*payloads.begin(), kPassingConversation);

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

TEST(QueryBehaviorTest, QueryReadsLimitedSetOfBlocksBasedOnSessionIdFilter) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildQueryConfig(base_dir);
  const std::string target_session = "target_session";
  const int expected_passing_records = 4;
  const absl::Time base_time = absl::Now() - absl::Minutes(25);
  const std::vector<QuerySeedRecord> records = BuildShuffledRecords(
      120, expected_passing_records, [](int i) { return absl::StrCat("agent_", i % 7); },
      [&](int i) {
        return i < expected_passing_records ? target_session
                                            : absl::StrCat("other_session_", 1 + (i % 8));
      },
      [&](int i) { return base_time + absl::Seconds(i); });
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  auto* metadata_store = engine.GetMetadataStoreForTest();

  EXPECT_TRUE(init_status.ok());
  EXPECT_NE(metadata_store, nullptr);
  EXPECT_EQ(records.size(), 120);
  EXPECT_TRUE(std::filesystem::exists(config.baked_block_dir));

  for (const auto& record : records) {
    const auto append_status = engine.AppendConversation(record.agent_id, record.session_id,
                                                         record.conversation_log, record.timestamp);
    EXPECT_TRUE(append_status.ok());
  }
  absl::SleepFor(absl::Seconds(1));

  kalki::QueryFilter all_filter;
  all_filter.caller_agent_id = "query_tester";
  const auto all_baked_or = metadata_store->FindCandidateBakedBlocks(all_filter);
  const auto all_fresh_or = metadata_store->FindCandidateFreshBlocks(all_filter);
  kalki::QueryFilter filtered;
  filtered.caller_agent_id = "query_tester";
  filtered.session_id = target_session;
  const auto filtered_baked_or = metadata_store->FindCandidateBakedBlocks(filtered);
  const auto filtered_fresh_or = metadata_store->FindCandidateFreshBlocks(filtered);

  const int all_block_count = all_baked_or.ok() && all_fresh_or.ok()
                                  ? static_cast<int>(all_baked_or->size() + all_fresh_or->size())
                                  : 0;
  const int filtered_block_count =
      filtered_baked_or.ok() && filtered_fresh_or.ok()
          ? static_cast<int>(filtered_baked_or->size() + filtered_fresh_or->size())
          : 0;

  EXPECT_TRUE(all_baked_or.ok());
  EXPECT_TRUE(all_fresh_or.ok());
  EXPECT_TRUE(filtered_baked_or.ok());
  EXPECT_TRUE(filtered_fresh_or.ok());
  EXPECT_GT(all_block_count, 0);
  EXPECT_GT(filtered_block_count, 0);
  EXPECT_LT(filtered_block_count, all_block_count);

  const auto query_or = engine.QueryLogs(kQueryString, filtered);

  std::set<std::string> payloads;
  if (query_or.ok()) {
    for (const auto& record : query_or->records) {
      payloads.insert(record.raw_conversation_log);
    }
  }
  EXPECT_TRUE(query_or.ok());
  EXPECT_EQ(static_cast<int>(query_or->records.size()), expected_passing_records);
  EXPECT_EQ(payloads.size(), 1u);
  EXPECT_EQ(*payloads.begin(), kPassingConversation);

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

TEST(QueryBehaviorTest, QueryReadsLimitedSetOfBlocksBasedOnTimestampIdFilter) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildQueryConfig(base_dir);
  const int expected_passing_records = 3;
  const absl::Time base_time = absl::Now() - absl::Minutes(40);
  const absl::Time passing_time = base_time + absl::Hours(24);
  const std::vector<QuerySeedRecord> records = BuildShuffledRecords(
      120, expected_passing_records, [](int i) { return absl::StrCat("agent_", i % 6); },
      [](int i) { return absl::StrCat("session_", i % 9); },
      [&](int i) {
        return i < expected_passing_records ? passing_time : base_time + absl::Seconds(i);
      });
  kalki::DatabaseEngine engine(config, std::make_unique<kalki::test::FakeLlmClient>(),
                               std::make_unique<kalki::test::FakeEmbeddingClient>());
  const auto init_status = engine.Initialize();
  auto* metadata_store = engine.GetMetadataStoreForTest();

  EXPECT_TRUE(init_status.ok());
  EXPECT_NE(metadata_store, nullptr);
  EXPECT_EQ(records.size(), 120);
  EXPECT_TRUE(std::filesystem::exists(config.baked_block_dir));

  for (const auto& record : records) {
    const auto append_status = engine.AppendConversation(record.agent_id, record.session_id,
                                                         record.conversation_log, record.timestamp);
    EXPECT_TRUE(append_status.ok());
  }
  absl::SleepFor(absl::Seconds(1));

  kalki::QueryFilter all_filter;
  all_filter.caller_agent_id = "query_tester";
  const auto all_baked_or = metadata_store->FindCandidateBakedBlocks(all_filter);
  const auto all_fresh_or = metadata_store->FindCandidateFreshBlocks(all_filter);
  kalki::QueryFilter filtered;
  filtered.caller_agent_id = "query_tester";
  filtered.start_time = passing_time - absl::Minutes(1);
  filtered.end_time = passing_time + absl::Minutes(1);
  const auto filtered_baked_or = metadata_store->FindCandidateBakedBlocks(filtered);
  const auto filtered_fresh_or = metadata_store->FindCandidateFreshBlocks(filtered);

  const int all_block_count = all_baked_or.ok() && all_fresh_or.ok()
                                  ? static_cast<int>(all_baked_or->size() + all_fresh_or->size())
                                  : 0;
  const int filtered_block_count =
      filtered_baked_or.ok() && filtered_fresh_or.ok()
          ? static_cast<int>(filtered_baked_or->size() + filtered_fresh_or->size())
          : 0;

  EXPECT_TRUE(all_baked_or.ok());
  EXPECT_TRUE(all_fresh_or.ok());
  EXPECT_TRUE(filtered_baked_or.ok());
  EXPECT_TRUE(filtered_fresh_or.ok());
  EXPECT_GT(all_block_count, 0);
  EXPECT_GT(filtered_block_count, 0);
  EXPECT_LT(filtered_block_count, all_block_count);

  const auto query_or = engine.QueryLogs(kQueryString, filtered);

  std::set<std::string> payloads;
  if (query_or.ok()) {
    for (const auto& record : query_or->records) {
      payloads.insert(record.raw_conversation_log);
    }
  }
  EXPECT_TRUE(query_or.ok());
  EXPECT_EQ(static_cast<int>(query_or->records.size()), expected_passing_records);
  EXPECT_EQ(payloads.size(), 1u);
  EXPECT_EQ(*payloads.begin(), kPassingConversation);

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

}  // namespace
