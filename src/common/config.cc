#include "kalki/common/config.h"

#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

ABSL_FLAG(std::string, data_dir, "./data", "Database data directory");
ABSL_FLAG(std::string, wal_path, "./data/wal.log", "WAL file path");
ABSL_FLAG(std::string, metadata_db_path, "./data/metadata.db", "Metadata SQLite DB path");
ABSL_FLAG(std::string, fresh_block_dir, "./data/blocks/fresh", "Fresh block directory");
ABSL_FLAG(std::string, baked_block_dir, "./data/blocks/baked", "Baked block directory");
ABSL_FLAG(std::string, grpc_listen_address, "0.0.0.0:8080", "gRPC listen address");
ABSL_FLAG(std::string, llm_api_key, "", "LLM API key");
ABSL_FLAG(std::string, llm_model, "gemini-1.5-flash", "Gemini model name");
ABSL_FLAG(int, max_records_per_fresh_block, 2000, "Max records per fresh block");
ABSL_FLAG(int, wal_read_batch_size, 200, "WAL read batch size");
ABSL_FLAG(int, query_worker_threads, 8, "Worker thread count for query execution");
ABSL_FLAG(double, similarity_threshold, 0.2, "Similarity threshold for summary filtering");
ABSL_FLAG(int, query_timeout_ms, 250, "Coordinator timeout in milliseconds");
ABSL_FLAG(int, worker_timeout_ms, 50, "Per-worker scheduling timeout in milliseconds");
ABSL_FLAG(int, summary_chunk_bytes, 2048, "Chunk size used for LLM summarization");
ABSL_FLAG(int, ingestion_poll_interval_ms, 50, "Ingestion polling interval in milliseconds");
ABSL_FLAG(int, compaction_poll_interval_ms, 100, "Compaction polling interval in milliseconds");

namespace kalki {

DatabaseConfig LoadConfigFromFlags() {
  DatabaseConfig cfg;
  cfg.data_dir = absl::GetFlag(FLAGS_data_dir);
  cfg.wal_path = absl::GetFlag(FLAGS_wal_path);
  cfg.metadata_db_path = absl::GetFlag(FLAGS_metadata_db_path);
  cfg.fresh_block_dir = absl::GetFlag(FLAGS_fresh_block_dir);
  cfg.baked_block_dir = absl::GetFlag(FLAGS_baked_block_dir);
  cfg.grpc_listen_address = absl::GetFlag(FLAGS_grpc_listen_address);

  cfg.llm_api_key = absl::GetFlag(FLAGS_llm_api_key);
  cfg.llm_model = absl::GetFlag(FLAGS_llm_model);

  cfg.max_records_per_fresh_block = absl::GetFlag(FLAGS_max_records_per_fresh_block);
  cfg.wal_read_batch_size = absl::GetFlag(FLAGS_wal_read_batch_size);
  cfg.query_worker_threads = absl::GetFlag(FLAGS_query_worker_threads);

  cfg.similarity_threshold = absl::GetFlag(FLAGS_similarity_threshold);
  cfg.query_timeout_ms = absl::GetFlag(FLAGS_query_timeout_ms);
  cfg.worker_timeout_ms = absl::GetFlag(FLAGS_worker_timeout_ms);

  cfg.summary_chunk_bytes = static_cast<size_t>(absl::GetFlag(FLAGS_summary_chunk_bytes));
  cfg.ingestion_poll_interval = absl::Milliseconds(absl::GetFlag(FLAGS_ingestion_poll_interval_ms));
  cfg.compaction_poll_interval =
      absl::Milliseconds(absl::GetFlag(FLAGS_compaction_poll_interval_ms));
  return cfg;
}

}  // namespace kalki
