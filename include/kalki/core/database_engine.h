#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "kalki/common/config.h"
#include "kalki/common/task_queue.h"
#include "kalki/common/thread_pool.h"
#include "kalki/common/types.h"

namespace kalki {

class WalStore;
class MetadataStore;
class EmbeddingClient;
class IngestionWorker;
class CompactionWorker;
class QueryCoordinator;

class DatabaseEngine {
 public:
  explicit DatabaseEngine(DatabaseConfig config);
  DatabaseEngine(DatabaseConfig config, std::unique_ptr<EmbeddingClient> embedding_client);
  ~DatabaseEngine();

  absl::Status Initialize();
  void Shutdown();

  absl::Status AppendConversation(const std::string& agent_id, const std::string& session_id,
                                  const std::string& conversation_log, absl::Time timestamp,
                                  std::optional<std::string> summary = std::nullopt);

  absl::StatusOr<QueryExecutionResult> QueryLogs(const std::string& query,
                                                 const QueryFilter& filter);

  WalStore* GetWalStoreForTest();
  MetadataStore* GetMetadataStoreForTest();

 private:
  void IngestionLoop();
  void CompactionLoop();

  DatabaseConfig config_;
  std::atomic<bool> stop_{false};

  TaskQueue<CompactionTask> compaction_queue_;
  ThreadPool query_thread_pool_;

  std::unique_ptr<WalStore> wal_store_;
  std::unique_ptr<MetadataStore> metadata_store_;
  std::unique_ptr<EmbeddingClient> embedding_client_;
  std::unique_ptr<IngestionWorker> ingestion_worker_;
  std::unique_ptr<CompactionWorker> compaction_worker_;
  std::unique_ptr<QueryCoordinator> query_coordinator_;

  std::thread ingestion_thread_;
  std::thread compaction_thread_;
};

}  // namespace kalki
