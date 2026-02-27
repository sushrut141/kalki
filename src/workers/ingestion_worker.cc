#include "kalki/workers/ingestion_worker.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"

namespace kalki {

namespace {

int64_t ToMicros(const google::protobuf::Timestamp& ts) {
  return static_cast<int64_t>(ts.seconds()) * 1000000LL + static_cast<int64_t>(ts.nanos()) / 1000LL;
}

}  // namespace

IngestionWorker::IngestionWorker(const DatabaseConfig& config, WalStore* wal_store,
                                 MetadataStore* metadata_store, LlmClient* llm_client,
                                 EmbeddingClient* embedding_client,
                                 TaskQueue<CompactionTask>* compaction_queue)
    : config_(config),
      wal_store_(wal_store),
      metadata_store_(metadata_store),
      llm_client_(llm_client),
      embedding_client_(embedding_client),
      compaction_queue_(compaction_queue) {}

absl::Status IngestionWorker::EnsureActiveFreshBlock() {
  if (active_block_id_ != 0) {
    return absl::OkStatus();
  }

  auto active_or = metadata_store_->GetActiveFreshBlock();
  if (!active_or.ok()) {
    return active_or.status();
  }

  if (active_or->has_value()) {
    const BlockMetadata& block = active_or->value();
    active_block_id_ = block.block_id;
    active_block_path_ = block.block_path;
    active_record_count_ = block.record_count;
    active_min_ts_micros_ =
        block.record_count == 0 ? std::numeric_limits<int64_t>::max() : block.min_timestamp_micros;
    active_max_ts_micros_ =
        block.record_count == 0 ? std::numeric_limits<int64_t>::min() : block.max_timestamp_micros;

    auto open_status = active_writer_.Open(active_block_path_);
    if (!open_status.ok()) {
      return open_status;
    }
    active_writer_open_ = true;
    return absl::OkStatus();
  }

  // No active fresh block exists so create one.
  const int64_t now = absl::ToUnixMicros(absl::Now());
  const std::string path = absl::StrCat(config_.fresh_block_dir, "/fresh_", now, ".fblk");
  auto id_or = metadata_store_->CreateFreshBlock(path);
  if (!id_or.ok()) {
    return id_or.status();
  }
  auto open_status = active_writer_.Open(path);
  if (!open_status.ok()) {
    return open_status;
  }
  active_writer_open_ = true;

  active_block_id_ = *id_or;
  active_block_path_ = path;
  active_record_count_ = 0;
  active_min_ts_micros_ = std::numeric_limits<int64_t>::max();
  active_max_ts_micros_ = std::numeric_limits<int64_t>::min();

  LOG(INFO) << "component=ingestion event=fresh_block_created block_id=" << active_block_id_
            << " path=" << active_block_path_;
  return absl::OkStatus();
}

absl::Status IngestionWorker::RotateFreshBlock() {
  if (active_block_id_ == 0) {
    return absl::OkStatus();
  }

  int64_t min_ts = active_record_count_ > 0 ? active_min_ts_micros_ : 0;
  int64_t max_ts = active_record_count_ > 0 ? active_max_ts_micros_ : 0;

  auto seal_status =
      metadata_store_->SealFreshBlock(active_block_id_, active_record_count_, min_ts, max_ts);
  CHECK(seal_status.ok()) << "ingestion failed to seal fresh block id=" << active_block_id_
                          << " status=" << seal_status;

  compaction_queue_->Push(
      CompactionTask{.fresh_block_id = active_block_id_, .fresh_block_path = active_block_path_});

  LOG(INFO) << "component=ingestion event=fresh_block_sealed block_id=" << active_block_id_
            << " records=" << active_record_count_;

  active_writer_.Close();
  active_writer_open_ = false;
  active_block_id_ = 0;
  active_block_path_.clear();
  active_record_count_ = 0;
  active_min_ts_micros_ = std::numeric_limits<int64_t>::max();
  active_max_ts_micros_ = std::numeric_limits<int64_t>::min();

  return EnsureActiveFreshBlock();
}

absl::Status IngestionWorker::MaybeTrimWal(int64_t current_offset, int64_t wal_record_count) {
  const int keep_records = std::max(1, config_.max_records_per_fresh_block);
  const int threshold_records = config_.wal_trim_threshold_records > 0
                                    ? config_.wal_trim_threshold_records
                                    : 2 * keep_records;

  if (wal_record_count <= threshold_records) {
    return absl::OkStatus();
  }

  auto lock_status = wal_store_->LockForMaintenance();
  if (!lock_status.ok()) {
    return lock_status;
  }
  auto unlock = absl::Cleanup([this]() { wal_store_->UnlockForMaintenance(); });

  auto trim_or = wal_store_->TrimToLastRecordsLocked(keep_records, current_offset);
  if (!trim_or.ok()) {
    return trim_or.status();
  }

  auto st = metadata_store_->SetWalOffset(trim_or->new_offset);
  CHECK(st.ok()) << "ingestion failed to update metadata wal offset after trim status=" << st;
  st = metadata_store_->SetWalRecordCount(trim_or->kept_records);
  CHECK(st.ok()) << "ingestion failed to update metadata wal count after trim status=" << st;

  LOG(INFO) << "component=ingestion event=wal_trimmed total_records=" << trim_or->total_records
            << " kept_records=" << trim_or->kept_records
            << " threshold_records=" << threshold_records << " new_offset=" << trim_or->new_offset;
  return absl::OkStatus();
}

absl::Status IngestionWorker::RunOnce() {
  auto status = EnsureActiveFreshBlock();
  if (!status.ok()) {
    return status;
  }

  auto offset_or = metadata_store_->GetWalOffset();
  if (!offset_or.ok()) {
    return offset_or.status();
  }
  auto wal_count_or = metadata_store_->GetWalRecordCount();
  if (!wal_count_or.ok()) {
    return wal_count_or.status();
  }

  auto batch_or = wal_store_->ReadBatchFromOffset(*offset_or, config_.wal_read_batch_size);
  if (!batch_or.ok()) {
    return batch_or.status();
  }
  if (batch_or->empty()) {
    return MaybeTrimWal(*offset_or, *wal_count_or);
  }

  int64_t latest_offset = *offset_or;
  for (const auto& envelope : *batch_or) {
    WalRecord wal_record;
    if (!wal_record.ParseFromString(envelope.payload)) {
      return absl::InternalError("failed parsing WAL protobuf record");
    }
    DLOG(INFO) << "component=ingestion event=wal_record_loaded offset=" << envelope.start_offset
               << " agent_id=" << wal_record.agent_id();

    auto summaries_or = llm_client_->SummarizeConversation(wal_record.conversation_log());
    if (!summaries_or.ok()) {
      return summaries_or.status();
    }

    for (const auto& summary : *summaries_or) {
      ProcessedRecord processed;
      processed.set_agent_id(wal_record.agent_id());
      processed.set_session_id(wal_record.session_id());
      *processed.mutable_timestamp() = wal_record.timestamp();
      processed.set_raw_conversation_log(wal_record.conversation_log());
      processed.set_summary(summary);
      auto embedding_or = embedding_client_->EmbedText(summary);
      if (!embedding_or.ok()) {
        return embedding_or.status();
      }
      for (float v : *embedding_or) {
        processed.add_summary_embedding(v);
      }

      CHECK(active_writer_open_) << "ingestion active writer is not initialized for block id="
                                 << active_block_id_;
      status = active_writer_.Append(processed);
      CHECK(status.ok()) << "ingestion failed to append processed record to fresh block id="
                         << active_block_id_ << " status=" << status;

      const int64_t ts_micros = ToMicros(wal_record.timestamp());
      active_min_ts_micros_ = std::min(active_min_ts_micros_, ts_micros);
      active_max_ts_micros_ = std::max(active_max_ts_micros_, ts_micros);
      ++active_record_count_;

      if (active_record_count_ >= config_.max_records_per_fresh_block) {
        status = RotateFreshBlock();
        CHECK(status.ok()) << "ingestion failed rotating fresh block status=" << status;
      }
    }
    latest_offset = envelope.next_offset;
  }

  if (active_block_id_ != 0) {
    status = metadata_store_->SetBlockRecordCount(active_block_id_, active_record_count_);
    CHECK(status.ok()) << "ingestion failed to persist active block record count block_id="
                       << active_block_id_ << " count=" << active_record_count_
                       << " status=" << status;
  }

  status = metadata_store_->SetWalOffset(latest_offset);
  CHECK(status.ok()) << "ingestion failed to update wal offset at end of batch offset="
                     << latest_offset << " status=" << status;

  status = MaybeTrimWal(latest_offset, *wal_count_or);
  CHECK(status.ok()) << "ingestion failed in wal trim check status=" << status;

  LOG(INFO) << "component=ingestion event=processed_batch records=" << batch_or->size();
  DLOG(INFO) << "component=ingestion event=active_block_state block_id=" << active_block_id_
             << " count=" << active_record_count_;
  return absl::OkStatus();
}

}  // namespace kalki
