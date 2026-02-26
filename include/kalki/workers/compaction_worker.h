#pragma once

#include <string>

#include "absl/status/status.h"
#include "kalki/common/config.h"
#include "kalki/common/task_queue.h"
#include "kalki/common/types.h"
#include "kalki/metadata/metadata_store.h"

namespace kalki {

class CompactionWorker {
 public:
  CompactionWorker(const DatabaseConfig& config, MetadataStore* metadata_store,
                   TaskQueue<CompactionTask>* compaction_queue);

  absl::Status RunOnce();

 private:
  DatabaseConfig config_;
  MetadataStore* metadata_store_;
  TaskQueue<CompactionTask>* compaction_queue_;
};

}  // namespace kalki
