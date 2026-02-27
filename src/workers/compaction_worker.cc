#include "kalki/workers/compaction_worker.h"

#include <filesystem>
#include <string>

#include "absl/log/check.h"
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
  // Block lifecycle modeled by (block_type, state):
  // - FRESH + ACTIVE: current append target for ingestion.
  // - FRESH + SEALED: closed for writes, eligible for compaction.
  // - FRESH + COMPACTED: already compacted into a baked block.
  // - BAKED + READY: queryable compacted block.
  //
  // This worker consumes sealed fresh blocks and creates baked ready blocks,
  // then marks the source fresh block compacted in metadata.
  auto maybe_task = compaction_queue_->PopWithTimeout(absl::Milliseconds(5));
  if (!maybe_task.has_value()) {
    DLOG(INFO) << "component=compaction event=no_task";
    return absl::OkStatus();
  }

  const CompactionTask task = std::move(*maybe_task);

  auto records_or = FreshBlockReader::ReadAll(task.fresh_block_path);
  CHECK(records_or.ok()) << "compaction failed to read fresh block fresh_block_id="
                         << task.fresh_block_id << " path=" << task.fresh_block_path
                         << " status=" << records_or.status();
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
  CHECK(bake_status.ok()) << "compaction failed to write baked block fresh_block_id="
                          << task.fresh_block_id << " baked_path=" << baked_path
                          << " status=" << bake_status;

  auto create_status = metadata_store_->CreateBakedBlock(
      task.fresh_block_id, baked_path, static_cast<int64_t>(records_or->size()), min_ts, max_ts,
      agent_bloom, session_bloom);
  CHECK(create_status.ok()) << "compaction failed to write baked metadata fresh_block_id="
                            << task.fresh_block_id << " baked_path=" << baked_path
                            << " status=" << create_status.status();

  LOG(INFO) << "component=compaction event=baked_block_created fresh_block_id="
            << task.fresh_block_id << " baked_block_id=" << *create_status
            << " records=" << records_or->size();

  std::error_code ec;
  const bool deleted = std::filesystem::remove(task.fresh_block_path, ec);
  if (!deleted || ec) {
    LOG(WARNING) << "component=compaction event=fresh_block_delete_failed fresh_block_id="
                 << task.fresh_block_id << " path=" << task.fresh_block_path
                 << " error=" << ec.message();
  } else {
    LOG(INFO) << "component=compaction event=fresh_block_deleted fresh_block_id="
              << task.fresh_block_id << " path=" << task.fresh_block_path;
  }

  return absl::OkStatus();
}

}  // namespace kalki
