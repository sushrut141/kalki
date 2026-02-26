#pragma once

#include <cstddef>
#include <string>

#include "absl/time/time.h"

namespace kalki {

struct DatabaseConfig {
  std::string data_dir;
  std::string wal_path;
  std::string metadata_db_path;
  std::string fresh_block_dir;
  std::string baked_block_dir;
  std::string grpc_listen_address;

  std::string llm_api_key;
  std::string llm_model;
  std::string embedding_model_path;
  int embedding_threads = 2;

  int max_records_per_fresh_block = 2000;
  int wal_trim_threshold_records = 0;
  int wal_read_batch_size = 200;
  int query_worker_threads = 8;

  double similarity_threshold = 0.2;
  int query_timeout_ms = 250;
  int worker_timeout_ms = 50;

  size_t summary_chunk_bytes = 2048;

  absl::Duration ingestion_poll_interval = absl::Milliseconds(50);
  absl::Duration compaction_poll_interval = absl::Milliseconds(100);
};

DatabaseConfig LoadConfigFromFlags();

}  // namespace kalki
