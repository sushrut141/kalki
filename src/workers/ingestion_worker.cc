#include "kalki/workers/ingestion_worker.h"

#include <filesystem>
#include <limits>
#include <string>
#include <utility>

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
                                 TaskQueue<CompactionTask>* compaction_queue)
    : config_(config),
      wal_store_(wal_store),
      metadata_store_(metadata_store),
      llm_client_(llm_client),
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
    return absl::OkStatus();
  }

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

  auto seal_status = metadata_store_->SealFreshBlock(active_block_id_, min_ts, max_ts);
  if (!seal_status.ok()) {
    return seal_status;
  }

  compaction_queue_->Push(
      CompactionTask{.fresh_block_id = active_block_id_, .fresh_block_path = active_block_path_});

  LOG(INFO) << "component=ingestion event=fresh_block_sealed block_id=" << active_block_id_
            << " records=" << active_record_count_;

  active_writer_.Close();
  active_block_id_ = 0;
  active_block_path_.clear();
  active_record_count_ = 0;
  active_min_ts_micros_ = std::numeric_limits<int64_t>::max();
  active_max_ts_micros_ = std::numeric_limits<int64_t>::min();

  return EnsureActiveFreshBlock();
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

  auto batch_or = wal_store_->ReadBatchFromOffset(*offset_or, config_.wal_read_batch_size);
  if (!batch_or.ok()) {
    return batch_or.status();
  }
  if (batch_or->empty()) {
    return absl::OkStatus();
  }

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

      status = active_writer_.Append(processed);
      if (!status.ok()) {
        return status;
      }

      status = metadata_store_->IncrementBlockRecordCount(active_block_id_, 1);
      if (!status.ok()) {
        return status;
      }

      const int64_t ts_micros = ToMicros(wal_record.timestamp());
      active_min_ts_micros_ = std::min(active_min_ts_micros_, ts_micros);
      active_max_ts_micros_ = std::max(active_max_ts_micros_, ts_micros);
      ++active_record_count_;

      if (active_record_count_ >= config_.max_records_per_fresh_block) {
        status = RotateFreshBlock();
        if (!status.ok()) {
          return status;
        }
      }
    }

    status = metadata_store_->SetWalOffset(envelope.next_offset);
    if (!status.ok()) {
      return status;
    }
  }

  LOG(INFO) << "component=ingestion event=processed_batch records=" << batch_or->size();
  DLOG(INFO) << "component=ingestion event=active_block_state block_id=" << active_block_id_
             << " count=" << active_record_count_;
  return absl::OkStatus();
}

}  // namespace kalki
