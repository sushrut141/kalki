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
  explicit WalStore(std::string wal_path);

  absl::Status Initialize() ABSL_LOCKS_EXCLUDED(mutex_);
  absl::StatusOr<int64_t> Append(const std::string& payload_bytes) ABSL_LOCKS_EXCLUDED(mutex_);
  absl::StatusOr<std::vector<WalEnvelope>> ReadBatchFromOffset(int64_t offset, int max_records)
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  std::string wal_path_;
  absl::Mutex mutex_;
};

}  // namespace kalki
