#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "kalki/common/types.h"

namespace kalki {

// Thread-safe write-ahead log store.
class WalStore {
 public:
  struct TrimResult {
    bool trimmed = false;
    int64_t total_records = 0;
    int64_t kept_records = 0;
    int64_t new_offset = 0;
  };

  explicit WalStore(std::string wal_path);

  absl::Status Initialize() ABSL_LOCKS_EXCLUDED(mutex_);
  absl::StatusOr<int64_t> Append(const std::string& payload_bytes) ABSL_LOCKS_EXCLUDED(mutex_);
  absl::StatusOr<std::vector<WalEnvelope>> ReadBatchFromOffset(int64_t offset, int max_records)
      ABSL_LOCKS_EXCLUDED(mutex_);
  absl::Status LockForMaintenance() ABSL_EXCLUSIVE_LOCK_FUNCTION(mutex_);
  void UnlockForMaintenance() ABSL_UNLOCK_FUNCTION(mutex_);
  absl::StatusOr<TrimResult> TrimToLastRecordsLocked(int keep_records, int64_t current_offset)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

 private:
  std::string wal_path_;
  absl::Mutex mutex_;
};

}  // namespace kalki
