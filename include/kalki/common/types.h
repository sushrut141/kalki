#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/time/time.h"

namespace kalki {

struct QueryFilter {
  std::string caller_agent_id;
  std::optional<std::string> session_id;
  std::optional<std::string> agent_id;
  std::optional<absl::Time> start_time;
  std::optional<absl::Time> end_time;
};

struct QueryRecord {
  std::string agent_id;
  std::string session_id;
  absl::Time timestamp;
  std::string raw_conversation_log;
};

enum class QueryCompletionStatus {
  kComplete,
  kIncomplete,
  kInternalError,
};

struct QueryExecutionResult {
  QueryCompletionStatus status = QueryCompletionStatus::kComplete;
  std::vector<QueryRecord> records;
  std::string error_message;
};

struct BlockMetadata {
  int64_t block_id = 0;
  std::string block_path;
  std::string block_type;
  std::string state;
  int64_t record_count = 0;
  int64_t min_timestamp_micros = 0;
  int64_t max_timestamp_micros = 0;
  std::string agent_bloom;
  std::string session_bloom;
  int64_t parent_block_id = 0;
};

struct WalEnvelope {
  int64_t start_offset = 0;
  int64_t next_offset = 0;
  std::string payload;
};

struct CompactionTask {
  int64_t fresh_block_id = 0;
  std::string fresh_block_path;
};

}  // namespace kalki
