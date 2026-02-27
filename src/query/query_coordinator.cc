#include "kalki/query/query_coordinator.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "kalki/storage/baked_block.h"
#include "kalki/storage/bloom_filter.h"
#include "kalki/storage/fresh_block.h"

namespace kalki {

namespace {

bool MatchesRecordFilter(const BakedHeaderEntry& entry, const QueryFilter& filter,
                         int64_t score_ts_micros) {
  if (filter.agent_id.has_value() && entry.agent_id() != *filter.agent_id) {
    return false;
  }
  if (filter.session_id.has_value() && entry.session_id() != *filter.session_id) {
    return false;
  }
  if (filter.start_time.has_value() && score_ts_micros < absl::ToUnixMicros(*filter.start_time)) {
    return false;
  }
  if (filter.end_time.has_value() && score_ts_micros > absl::ToUnixMicros(*filter.end_time)) {
    return false;
  }
  return true;
}

int64_t ToMicros(const google::protobuf::Timestamp& ts) {
  return static_cast<int64_t>(ts.seconds()) * 1000000LL + static_cast<int64_t>(ts.nanos()) / 1000LL;
}

bool MatchesProcessedRecordFilter(const ProcessedRecord& record, const QueryFilter& filter) {
  if (filter.agent_id.has_value() && record.agent_id() != *filter.agent_id) {
    return false;
  }
  if (filter.session_id.has_value() && record.session_id() != *filter.session_id) {
    return false;
  }

  const int64_t ts_micros = ToMicros(record.timestamp());
  if (filter.start_time.has_value() && ts_micros < absl::ToUnixMicros(*filter.start_time)) {
    return false;
  }
  if (filter.end_time.has_value() && ts_micros > absl::ToUnixMicros(*filter.end_time)) {
    return false;
  }
  return true;
}

}  // namespace

QueryCoordinator::QueryCoordinator(const DatabaseConfig& config, MetadataStore* metadata_store,
                                   EmbeddingClient* embedding_client, ThreadPool* thread_pool)
    : config_(config),
      metadata_store_(metadata_store),
      embedding_client_(embedding_client),
      thread_pool_(thread_pool),
      similarity_engine_() {}

absl::StatusOr<std::vector<QueryRecord>> QueryCoordinator::ProcessBlockGroup(
    const std::vector<float>& query_embedding, const QueryFilter& filter,
    const std::vector<BlockMetadata>& group) const {
  std::vector<QueryRecord> output;

  for (const auto& block : group) {
    DLOG(INFO) << "component=query_worker event=scan_block block_id=" << block.block_id
               << " path=" << block.block_path;
    if (block.block_type == "FRESH") {
      auto records_or = FreshBlockReader::ReadAll(block.block_path);
      if (!records_or.ok()) {
        LOG(ERROR) << "component=query_worker event=fresh_read_failed block_id=" << block.block_id
                   << " error=" << records_or.status();
        continue;
      }

      std::vector<SimilarityEngine::EmbeddingView> embeddings;
      std::vector<size_t> embedding_to_record_index;
      embeddings.reserve(records_or->size());
      embedding_to_record_index.reserve(records_or->size());
      for (size_t i = 0; i < records_or->size(); ++i) {
        const ProcessedRecord& record = (*records_or)[i];
        if (record.summary_embedding_size() == 0) {
          continue;
        }
        embeddings.push_back(SimilarityEngine::EmbeddingView{
            .data = record.summary_embedding().data(),
            .dims = static_cast<size_t>(record.summary_embedding_size())});
        embedding_to_record_index.push_back(i);
      }

      auto scores_or = similarity_engine_.ScoreEmbeddings(query_embedding, embeddings);
      if (!scores_or.ok()) {
        return scores_or.status();
      }

      for (size_t i = 0; i < scores_or->size(); ++i) {
        if ((*scores_or)[i] < config_.similarity_threshold) {
          continue;
        }
        const ProcessedRecord& record = (*records_or)[embedding_to_record_index[i]];
        if (!MatchesProcessedRecordFilter(record, filter)) {
          continue;
        }

        QueryRecord out;
        out.agent_id = record.agent_id();
        out.session_id = record.session_id();
        out.timestamp = absl::FromUnixMicros(ToMicros(record.timestamp()));
        out.raw_conversation_log = record.raw_conversation_log();
        output.push_back(std::move(out));
      }
      continue;
    }

    if (filter.agent_id.has_value() && !block.agent_bloom.empty()) {
      BloomFilter filter_agent = BloomFilter::Deserialize(block.agent_bloom);
      if (!filter_agent.PossiblyContains(*filter.agent_id)) {
        continue;
      }
    }
    if (filter.session_id.has_value() && !block.session_bloom.empty()) {
      BloomFilter filter_session = BloomFilter::Deserialize(block.session_bloom);
      if (!filter_session.PossiblyContains(*filter.session_id)) {
        continue;
      }
    }

    uint64_t data_section_offset = 0;
    auto header_or = BakedBlockReader::ReadHeaderOnly(block.block_path, &data_section_offset);
    if (!header_or.ok()) {
      LOG(ERROR) << "component=query_worker event=header_read_failed block_id=" << block.block_id
                 << " error=" << header_or.status();
      continue;
    }

    std::vector<SimilarityEngine::EmbeddingView> embeddings;
    std::vector<int> embedding_to_entry_index;
    embeddings.reserve(static_cast<size_t>(header_or->entries_size()));
    embedding_to_entry_index.reserve(static_cast<size_t>(header_or->entries_size()));
    for (int i = 0; i < header_or->entries_size(); ++i) {
      const BakedHeaderEntry& entry = header_or->entries(i);
      if (entry.summary_embedding_size() == 0) {
        continue;
      }
      embeddings.push_back(SimilarityEngine::EmbeddingView{
          .data = entry.summary_embedding().data(),
          .dims = static_cast<size_t>(entry.summary_embedding_size())});
      embedding_to_entry_index.push_back(i);
    }

    auto scores_or = similarity_engine_.ScoreEmbeddings(query_embedding, embeddings);
    if (!scores_or.ok()) {
      return scores_or.status();
    }

    for (size_t i = 0; i < scores_or->size(); ++i) {
      const BakedHeaderEntry& entry = header_or->entries(embedding_to_entry_index[i]);
      const float score = (*scores_or)[i];
      const int64_t ts_micros = entry.timestamp_unix_micros();

      if (score < config_.similarity_threshold) {
        continue;
      }
      if (!MatchesRecordFilter(entry, filter, ts_micros)) {
        continue;
      }

      auto rec_or = BakedBlockReader::ReadRecordAt(block.block_path, data_section_offset,
                                                   entry.data_offset(), entry.data_size());
      if (!rec_or.ok()) {
        LOG(ERROR) << "component=query_worker event=record_read_failed block_id=" << block.block_id
                   << " error=" << rec_or.status();
        continue;
      }

      QueryRecord q;
      q.agent_id = rec_or->agent_id();
      q.session_id = rec_or->session_id();
      q.timestamp = absl::FromUnixMicros(ts_micros);
      q.raw_conversation_log = rec_or->raw_conversation_log();
      output.push_back(std::move(q));
    }
  }

  return output;
}

absl::StatusOr<QueryExecutionResult> QueryCoordinator::Query(const std::string& query,
                                                             const QueryFilter& filter) {
  auto query_embedding_or = embedding_client_->EmbedText(query);
  if (!query_embedding_or.ok()) {
    return query_embedding_or.status();
  }

  auto baked_candidates_or = metadata_store_->FindCandidateBakedBlocks(filter);
  if (!baked_candidates_or.ok()) {
    return baked_candidates_or.status();
  }
  auto fresh_candidates_or = metadata_store_->FindCandidateFreshBlocks(filter);
  if (!fresh_candidates_or.ok()) {
    return fresh_candidates_or.status();
  }

  std::vector<BlockMetadata> candidates = *baked_candidates_or;
  candidates.insert(candidates.end(), fresh_candidates_or->begin(), fresh_candidates_or->end());
  if (candidates.empty()) {
    QueryExecutionResult empty;
    empty.status = QueryCompletionStatus::kComplete;
    return empty;
  }

  const int worker_count = std::max(1, config_.query_worker_threads);
  std::vector<std::vector<BlockMetadata>> groups(static_cast<size_t>(worker_count));
  for (size_t i = 0; i < candidates.size(); ++i) {
    groups[i % groups.size()].push_back(candidates[i]);
  }

  absl::Mutex result_mu;
  std::vector<QueryRecord> all_results;
  std::atomic<int> pending(0);
  std::atomic<int> failed_groups(0);
  int scheduled_groups = 0;

  const absl::Time begin = absl::Now();

  for (const auto& group : groups) {
    if (group.empty()) {
      continue;
    }
    ++scheduled_groups;
    pending.fetch_add(1, std::memory_order_relaxed);

    auto submit_status = thread_pool_->Submit([&, group]() {
      auto group_or = ProcessBlockGroup(*query_embedding_or, filter, group);
      if (!group_or.ok()) {
        failed_groups.fetch_add(1, std::memory_order_relaxed);
      } else {
        absl::MutexLock lock(&result_mu);
        all_results.insert(all_results.end(), group_or->begin(), group_or->end());
      }
      pending.fetch_sub(1, std::memory_order_relaxed);
    });

    if (!submit_status.ok()) {
      pending.fetch_sub(1, std::memory_order_relaxed);
      failed_groups.fetch_add(1, std::memory_order_relaxed);
      LOG(ERROR) << "component=query_coordinator event=schedule_failed status=" << submit_status;
    }
  }

  const absl::Time hard_deadline = begin + absl::Milliseconds(config_.query_timeout_ms);
  while (pending.load(std::memory_order_relaxed) > 0 && absl::Now() < hard_deadline) {
    absl::SleepFor(absl::Milliseconds(1));
  }

  QueryExecutionResult result;
  {
    absl::MutexLock lock(&result_mu);
    result.records = std::move(all_results);
  }

  const bool pending_after_deadline = pending.load(std::memory_order_relaxed) > 0;
  const bool exceeded_worker_window =
      absl::Now() - begin > absl::Milliseconds(config_.worker_timeout_ms);

  if (pending_after_deadline || exceeded_worker_window) {
    result.status = QueryCompletionStatus::kIncomplete;
  } else if (scheduled_groups > 0 &&
             failed_groups.load(std::memory_order_relaxed) == scheduled_groups &&
             result.records.empty()) {
    result.status = QueryCompletionStatus::kInternalError;
    result.error_message = "all query workers failed";
  } else {
    result.status = QueryCompletionStatus::kComplete;
  }

  LOG(INFO) << "component=query_coordinator event=query_complete"
            << " candidate_blocks=" << candidates.size() << " records=" << result.records.size()
            << " status=" << static_cast<int>(result.status);

  return result;
}

}  // namespace kalki
