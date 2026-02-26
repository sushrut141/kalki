#include "kalki/workers/compaction_worker.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "kalki/storage/baked_block.h"
#include "kalki/storage/fresh_block.h"

namespace kalki {

CompactionWorker::CompactionWorker(const DatabaseConfig& config, MetadataStore* metadata_store,
                                   TaskQueue<CompactionTask>* compaction_queue)
    : config_(config), metadata_store_(metadata_store), compaction_queue_(compaction_queue) {}

absl::Status CompactionWorker::RunOnce() {
  auto maybe_task = compaction_queue_->PopWithTimeout(absl::Milliseconds(5));
  if (!maybe_task.has_value()) {
    DLOG(INFO) << "component=compaction event=no_task";
    return absl::OkStatus();
  }

  const CompactionTask task = std::move(*maybe_task);

  auto records_or = FreshBlockReader::ReadAll(task.fresh_block_path);
  if (!records_or.ok()) {
    return records_or.status();
  }
  if (records_or->empty()) {
    LOG(INFO) << "component=compaction event=skip_empty_block fresh_block_id="
              << task.fresh_block_id;
    return absl::OkStatus();
  }

  const std::string baked_path =
      absl::StrCat(config_.baked_block_dir, "/baked_", task.fresh_block_id, "_",
                   absl::ToUnixMicros(absl::Now()), ".bblk");

  int64_t min_ts = 0;
  int64_t max_ts = 0;
  std::string agent_bloom;
  std::string session_bloom;
  auto bake_status = BakedBlockWriter::Write(baked_path, std::move(*records_or), &min_ts, &max_ts,
                                             &agent_bloom, &session_bloom);
  if (!bake_status.ok()) {
    return bake_status;
  }

  auto create_status = metadata_store_->CreateBakedBlock(
      task.fresh_block_id, baked_path, static_cast<int64_t>(records_or->size()), min_ts, max_ts,
      agent_bloom, session_bloom);
  if (!create_status.ok()) {
    return create_status.status();
  }

  LOG(INFO) << "component=compaction event=baked_block_created fresh_block_id="
            << task.fresh_block_id << " baked_block_id=" << *create_status
            << " records=" << records_or->size();

  return absl::OkStatus();
}

}  // namespace kalki
