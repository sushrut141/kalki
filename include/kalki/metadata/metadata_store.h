#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "kalki/common/types.h"

struct sqlite3;

namespace kalki {

// Thread-safe SQLite-backed metadata store.
class MetadataStore {
 public:
  explicit MetadataStore(std::string db_path);
  ~MetadataStore();

  MetadataStore(const MetadataStore&) = delete;
  MetadataStore& operator=(const MetadataStore&) = delete;

  absl::Status Initialize() ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<int64_t> GetWalOffset() ABSL_LOCKS_EXCLUDED(mutex_);
  absl::Status SetWalOffset(int64_t offset) ABSL_LOCKS_EXCLUDED(mutex_);
  absl::StatusOr<int64_t> GetWalRecordCount() ABSL_LOCKS_EXCLUDED(mutex_);
  absl::Status SetWalRecordCount(int64_t count) ABSL_LOCKS_EXCLUDED(mutex_);
  absl::Status IncrementWalRecordCount(int64_t delta) ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<std::optional<BlockMetadata>> GetActiveFreshBlock() ABSL_LOCKS_EXCLUDED(mutex_);
  absl::StatusOr<int64_t> CreateFreshBlock(const std::string& path) ABSL_LOCKS_EXCLUDED(mutex_);
  absl::Status SetFreshBlockRecordCount(int64_t block_id, int64_t count)
      ABSL_LOCKS_EXCLUDED(mutex_);
  absl::Status IncrementBlockRecordCount(int64_t block_id, int64_t delta)
      ABSL_LOCKS_EXCLUDED(mutex_);
  absl::Status SealFreshBlock(int64_t block_id, int64_t record_count, int64_t min_ts_micros,
                              int64_t max_ts_micros) ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<int64_t> CreateBakedBlock(int64_t parent_fresh_block_id,
                                           const std::string& baked_path, int64_t record_count,
                                           int64_t min_ts_micros, int64_t max_ts_micros,
                                           const std::string& agent_bloom,
                                           const std::string& session_bloom)
      ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<std::vector<BlockMetadata>> FindCandidateBakedBlocks(
      const QueryFilter& filter) const ABSL_LOCKS_EXCLUDED(mutex_);
  absl::StatusOr<std::vector<BlockMetadata>> FindCandidateFreshBlocks(
      const QueryFilter& filter) const ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  absl::Status EnsureOpen() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  std::string db_path_;
  mutable absl::Mutex mutex_;
  sqlite3* db_ ABSL_GUARDED_BY(mutex_) = nullptr;
};

}  // namespace kalki
