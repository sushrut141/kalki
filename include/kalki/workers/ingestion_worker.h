#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "kalki.pb.h"
#include "kalki/common/config.h"
#include "kalki/common/task_queue.h"
#include "kalki/common/types.h"
#include "kalki/llm/embedding_client.h"
#include "kalki/llm/llm_client.h"
#include "kalki/metadata/metadata_store.h"
#include "kalki/storage/fresh_block.h"
#include "kalki/storage/wal.h"

namespace kalki {

class IngestionWorker {
 public:
  IngestionWorker(const DatabaseConfig& config, WalStore* wal_store, MetadataStore* metadata_store,
                  LlmClient* llm_client, EmbeddingClient* embedding_client,
                  TaskQueue<CompactionTask>* compaction_queue);

  absl::Status RunOnce();

 private:
  absl::Status EnsureActiveFreshBlock();
  absl::Status RotateFreshBlock();
  absl::Status MaybeTrimWal(int64_t current_offset, int64_t wal_record_count);

  DatabaseConfig config_;
  WalStore* wal_store_;
  MetadataStore* metadata_store_;
  LlmClient* llm_client_;
  EmbeddingClient* embedding_client_;
  TaskQueue<CompactionTask>* compaction_queue_;

  FreshBlockWriter active_writer_;
  bool active_writer_open_ = false;
  int64_t active_block_id_ = 0;
  std::string active_block_path_;
  int64_t active_record_count_ = 0;
  int64_t active_min_ts_micros_ = std::numeric_limits<int64_t>::max();
  int64_t active_max_ts_micros_ = std::numeric_limits<int64_t>::min();
};

}  // namespace kalki
