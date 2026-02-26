#include "kalki/core/database_engine.h"

#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "kalki.pb.h"
#include "kalki/llm/gemini_client.h"
#include "kalki/metadata/metadata_store.h"
#include "kalki/query/query_coordinator.h"
#include "kalki/storage/wal.h"
#include "kalki/workers/compaction_worker.h"
#include "kalki/workers/ingestion_worker.h"

namespace kalki {

namespace {

google::protobuf::Timestamp ToProtoTimestamp(absl::Time time) {
  const int64_t micros = absl::ToUnixMicros(time);
  google::protobuf::Timestamp ts;
  ts.set_seconds(micros / 1000000LL);
  ts.set_nanos(static_cast<int32_t>((micros % 1000000LL) * 1000));
  return ts;
}

}  // namespace

DatabaseEngine::DatabaseEngine(DatabaseConfig config)
    : config_(std::move(config)), query_thread_pool_(config_.query_worker_threads) {}

DatabaseEngine::~DatabaseEngine() { Shutdown(); }

absl::Status DatabaseEngine::Initialize() {
  std::filesystem::create_directories(config_.data_dir);
  std::filesystem::create_directories(config_.fresh_block_dir);
  std::filesystem::create_directories(config_.baked_block_dir);

  wal_store_ = std::make_unique<WalStore>(config_.wal_path);
  metadata_store_ = std::make_unique<MetadataStore>(config_.metadata_db_path);
  llm_client_ = std::make_unique<GeminiLlmClient>(config_.llm_api_key, config_.llm_model);

  if (auto st = wal_store_->Initialize(); !st.ok()) {
    return st;
  }
  if (auto st = metadata_store_->Initialize(); !st.ok()) {
    return st;
  }
  if (auto st = query_thread_pool_.Start(); !st.ok()) {
    return st;
  }

  ingestion_worker_ = std::make_unique<IngestionWorker>(
      config_, wal_store_.get(), metadata_store_.get(), llm_client_.get(), &compaction_queue_);
  compaction_worker_ =
      std::make_unique<CompactionWorker>(config_, metadata_store_.get(), &compaction_queue_);
  query_coordinator_ =
      std::make_unique<QueryCoordinator>(config_, metadata_store_.get(), &query_thread_pool_);

  stop_.store(false, std::memory_order_relaxed);
  ingestion_thread_ = std::thread([this]() { IngestionLoop(); });
  compaction_thread_ = std::thread([this]() { CompactionLoop(); });

  LOG(INFO) << "component=database_engine event=initialized";
  return absl::OkStatus();
}

void DatabaseEngine::Shutdown() {
  if (stop_.exchange(true, std::memory_order_relaxed)) {
    return;
  }

  if (ingestion_thread_.joinable()) {
    ingestion_thread_.join();
  }
  if (compaction_thread_.joinable()) {
    compaction_thread_.join();
  }
  query_thread_pool_.Stop();

  LOG(INFO) << "component=database_engine event=shutdown";
}

absl::Status DatabaseEngine::AppendConversation(const std::string& agent_id,
                                                const std::string& session_id,
                                                const std::string& conversation_log,
                                                absl::Time timestamp) {
  WalRecord record;
  record.set_agent_id(agent_id);
  record.set_session_id(session_id);
  record.set_conversation_log(conversation_log);
  *record.mutable_timestamp() = ToProtoTimestamp(timestamp);

  std::string payload;
  if (!record.SerializeToString(&payload)) {
    return absl::InternalError("failed serializing WAL record");
  }

  auto append_or = wal_store_->Append(payload);
  if (!append_or.ok()) {
    return append_or.status();
  }

  LOG(INFO) << "component=database_engine event=append_wal offset=" << *append_or
            << " agent_id=" << agent_id << " session_id=" << session_id;
  return absl::OkStatus();
}

absl::StatusOr<QueryExecutionResult> DatabaseEngine::QueryLogs(const std::string& query,
                                                               const QueryFilter& filter) {
  return query_coordinator_->Query(query, filter);
}

void DatabaseEngine::IngestionLoop() {
  while (!stop_.load(std::memory_order_relaxed)) {
    DLOG(INFO) << "component=database_engine event=ingestion_tick";
    auto st = ingestion_worker_->RunOnce();
    if (!st.ok()) {
      LOG(ERROR) << "component=ingestion event=run_once_failed status=" << st;
    }
    absl::SleepFor(config_.ingestion_poll_interval);
  }
}

void DatabaseEngine::CompactionLoop() {
  while (!stop_.load(std::memory_order_relaxed)) {
    DLOG(INFO) << "component=database_engine event=compaction_tick";
    auto st = compaction_worker_->RunOnce();
    if (!st.ok()) {
      LOG(ERROR) << "component=compaction event=run_once_failed status=" << st;
    }
    absl::SleepFor(config_.compaction_poll_interval);
  }
}

}  // namespace kalki
