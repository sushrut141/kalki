#include "kalki/metadata/metadata_store.h"

#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "sqlite3.h"

namespace kalki {

namespace {

absl::Status SqliteError(sqlite3* db, const std::string& message) {
  return absl::InternalError(absl::StrCat(message, ": ", sqlite3_errmsg(db)));
}

}  // namespace

MetadataStore::MetadataStore(std::string db_path) : db_path_(std::move(db_path)) {}

MetadataStore::~MetadataStore() {
  if (db_ != nullptr) {
    sqlite3_close(db_);
    db_ = nullptr;
  }
}

absl::Status MetadataStore::Initialize() {
  absl::MutexLock lock(&mutex_);
  std::filesystem::create_directories(std::filesystem::path(db_path_).parent_path());
  if (sqlite3_open(db_path_.c_str(), &db_) != SQLITE_OK) {
    return SqliteError(db_, "failed to open metadata sqlite db");
  }

  const char* kSchema = R"sql(
CREATE TABLE IF NOT EXISTS wal_state (
  id INTEGER PRIMARY KEY CHECK(id=1),
  last_processed_offset INTEGER NOT NULL
);
INSERT OR IGNORE INTO wal_state(id, last_processed_offset) VALUES (1, 0);

CREATE TABLE IF NOT EXISTS blocks (
  block_id INTEGER PRIMARY KEY AUTOINCREMENT,
  block_path TEXT NOT NULL,
  block_type TEXT NOT NULL,
  state TEXT NOT NULL,
  record_count INTEGER NOT NULL DEFAULT 0,
  min_timestamp_micros INTEGER NOT NULL DEFAULT 0,
  max_timestamp_micros INTEGER NOT NULL DEFAULT 0,
  agent_bloom BLOB,
  session_bloom BLOB,
  parent_block_id INTEGER NOT NULL DEFAULT 0,
  created_at_micros INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_blocks_range
ON blocks(block_type, state, min_timestamp_micros, max_timestamp_micros);
)sql";

  if (sqlite3_exec(db_, kSchema, nullptr, nullptr, nullptr) != SQLITE_OK) {
    return SqliteError(db_, "failed applying metadata schema");
  }

  LOG(INFO) << "component=metadata event=initialized path=" << db_path_;
  return absl::OkStatus();
}

absl::Status MetadataStore::EnsureOpen() const {
  if (db_ == nullptr) {
    return absl::FailedPreconditionError("metadata db is not open");
  }
  return absl::OkStatus();
}

absl::StatusOr<int64_t> MetadataStore::GetWalOffset() {
  absl::MutexLock lock(&mutex_);
  if (auto status = EnsureOpen(); !status.ok()) {
    return status;
  }

  sqlite3_stmt* stmt = nullptr;
  const char* sql = "SELECT last_processed_offset FROM wal_state WHERE id=1";
  if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    return SqliteError(db_, "failed preparing wal offset query");
  }

  int64_t offset = 0;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    offset = sqlite3_column_int64(stmt, 0);
  }
  sqlite3_finalize(stmt);
  return offset;
}

absl::Status MetadataStore::SetWalOffset(int64_t offset) {
  absl::MutexLock lock(&mutex_);
  if (auto status = EnsureOpen(); !status.ok()) {
    return status;
  }

  sqlite3_stmt* stmt = nullptr;
  const char* sql = "UPDATE wal_state SET last_processed_offset=? WHERE id=1";
  if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    return SqliteError(db_, "failed preparing wal offset update");
  }
  sqlite3_bind_int64(stmt, 1, offset);

  if (sqlite3_step(stmt) != SQLITE_DONE) {
    sqlite3_finalize(stmt);
    return SqliteError(db_, "failed updating wal offset");
  }
  sqlite3_finalize(stmt);
  DLOG(INFO) << "component=metadata event=wal_offset_updated offset=" << offset;
  return absl::OkStatus();
}

absl::StatusOr<std::optional<BlockMetadata>> MetadataStore::GetActiveFreshBlock() {
  absl::MutexLock lock(&mutex_);
  if (auto status = EnsureOpen(); !status.ok()) {
    return status;
  }

  sqlite3_stmt* stmt = nullptr;
  const char* sql =
      "SELECT block_id, block_path, block_type, state, record_count, "
      "min_timestamp_micros, max_timestamp_micros, parent_block_id "
      "FROM blocks WHERE block_type='FRESH' AND state='ACTIVE' "
      "ORDER BY block_id DESC LIMIT 1";
  if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    return SqliteError(db_, "failed preparing active fresh block query");
  }

  std::optional<BlockMetadata> out;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    BlockMetadata meta;
    meta.block_id = sqlite3_column_int64(stmt, 0);
    meta.block_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
    meta.block_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
    meta.state = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
    meta.record_count = sqlite3_column_int64(stmt, 4);
    meta.min_timestamp_micros = sqlite3_column_int64(stmt, 5);
    meta.max_timestamp_micros = sqlite3_column_int64(stmt, 6);
    meta.parent_block_id = sqlite3_column_int64(stmt, 7);
    out = std::move(meta);
  }
  sqlite3_finalize(stmt);
  return out;
}

absl::StatusOr<int64_t> MetadataStore::CreateFreshBlock(const std::string& path) {
  absl::MutexLock lock(&mutex_);
  if (auto status = EnsureOpen(); !status.ok()) {
    return status;
  }
  sqlite3_stmt* stmt = nullptr;
  const char* sql =
      "INSERT INTO blocks(block_path, block_type, state, record_count, created_at_micros) "
      "VALUES (?, 'FRESH', 'ACTIVE', 0, ?)";
  if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    return SqliteError(db_, "failed preparing create fresh block statement");
  }
  sqlite3_bind_text(stmt, 1, path.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 2, absl::ToUnixMicros(absl::Now()));

  if (sqlite3_step(stmt) != SQLITE_DONE) {
    sqlite3_finalize(stmt);
    return SqliteError(db_, "failed creating fresh block row");
  }
  sqlite3_finalize(stmt);
  return sqlite3_last_insert_rowid(db_);
}

absl::Status MetadataStore::IncrementBlockRecordCount(int64_t block_id, int64_t delta) {
  absl::MutexLock lock(&mutex_);
  if (auto status = EnsureOpen(); !status.ok()) {
    return status;
  }

  sqlite3_stmt* stmt = nullptr;
  const char* sql = "UPDATE blocks SET record_count = record_count + ? WHERE block_id = ?";
  if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    return SqliteError(db_, "failed preparing increment block statement");
  }
  sqlite3_bind_int64(stmt, 1, delta);
  sqlite3_bind_int64(stmt, 2, block_id);

  if (sqlite3_step(stmt) != SQLITE_DONE) {
    sqlite3_finalize(stmt);
    return SqliteError(db_, "failed incrementing block record count");
  }
  sqlite3_finalize(stmt);
  return absl::OkStatus();
}

absl::Status MetadataStore::SealFreshBlock(int64_t block_id, int64_t min_ts_micros,
                                           int64_t max_ts_micros) {
  absl::MutexLock lock(&mutex_);
  if (auto status = EnsureOpen(); !status.ok()) {
    return status;
  }

  sqlite3_stmt* stmt = nullptr;
  const char* sql =
      "UPDATE blocks SET state='SEALED', min_timestamp_micros=?, max_timestamp_micros=? "
      "WHERE block_id=?";
  if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    return SqliteError(db_, "failed preparing seal fresh block statement");
  }
  sqlite3_bind_int64(stmt, 1, min_ts_micros);
  sqlite3_bind_int64(stmt, 2, max_ts_micros);
  sqlite3_bind_int64(stmt, 3, block_id);

  if (sqlite3_step(stmt) != SQLITE_DONE) {
    sqlite3_finalize(stmt);
    return SqliteError(db_, "failed sealing fresh block");
  }
  sqlite3_finalize(stmt);
  return absl::OkStatus();
}

absl::StatusOr<int64_t> MetadataStore::CreateBakedBlock(int64_t parent_fresh_block_id,
                                                        const std::string& baked_path,
                                                        int64_t record_count, int64_t min_ts_micros,
                                                        int64_t max_ts_micros,
                                                        const std::string& agent_bloom,
                                                        const std::string& session_bloom) {
  absl::MutexLock lock(&mutex_);
  if (auto status = EnsureOpen(); !status.ok()) {
    return status;
  }

  sqlite3_stmt* stmt = nullptr;
  const char* insert_sql =
      "INSERT INTO blocks(block_path, block_type, state, record_count, min_timestamp_micros, "
      "max_timestamp_micros, agent_bloom, session_bloom, parent_block_id, created_at_micros) "
      "VALUES (?, 'BAKED', 'READY', ?, ?, ?, ?, ?, ?, ?)";
  if (sqlite3_prepare_v2(db_, insert_sql, -1, &stmt, nullptr) != SQLITE_OK) {
    return SqliteError(db_, "failed preparing create baked block statement");
  }
  sqlite3_bind_text(stmt, 1, baked_path.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 2, record_count);
  sqlite3_bind_int64(stmt, 3, min_ts_micros);
  sqlite3_bind_int64(stmt, 4, max_ts_micros);
  sqlite3_bind_blob(stmt, 5, agent_bloom.data(), static_cast<int>(agent_bloom.size()),
                    SQLITE_TRANSIENT);
  sqlite3_bind_blob(stmt, 6, session_bloom.data(), static_cast<int>(session_bloom.size()),
                    SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 7, parent_fresh_block_id);
  sqlite3_bind_int64(stmt, 8, absl::ToUnixMicros(absl::Now()));

  if (sqlite3_step(stmt) != SQLITE_DONE) {
    sqlite3_finalize(stmt);
    return SqliteError(db_, "failed inserting baked block metadata");
  }
  sqlite3_finalize(stmt);

  sqlite3_stmt* update_stmt = nullptr;
  const char* update_sql = "UPDATE blocks SET state='COMPACTED' WHERE block_id=?";
  if (sqlite3_prepare_v2(db_, update_sql, -1, &update_stmt, nullptr) != SQLITE_OK) {
    return SqliteError(db_, "failed preparing fresh compacted update statement");
  }
  sqlite3_bind_int64(update_stmt, 1, parent_fresh_block_id);
  if (sqlite3_step(update_stmt) != SQLITE_DONE) {
    sqlite3_finalize(update_stmt);
    return SqliteError(db_, "failed marking fresh block as compacted");
  }
  sqlite3_finalize(update_stmt);

  return sqlite3_last_insert_rowid(db_);
}

absl::StatusOr<std::vector<BlockMetadata>> MetadataStore::FindCandidateBakedBlocks(
    const QueryFilter& filter) const {
  absl::MutexLock lock(&mutex_);
  if (auto status = EnsureOpen(); !status.ok()) {
    return status;
  }

  int64_t min_bound = 0;
  int64_t max_bound = std::numeric_limits<int64_t>::max();
  if (filter.start_time.has_value()) {
    min_bound = absl::ToUnixMicros(*filter.start_time);
  }
  if (filter.end_time.has_value()) {
    max_bound = absl::ToUnixMicros(*filter.end_time);
  }

  sqlite3_stmt* stmt = nullptr;
  const char* sql =
      "SELECT block_id, block_path, block_type, state, record_count, "
      "min_timestamp_micros, max_timestamp_micros, agent_bloom, session_bloom, parent_block_id "
      "FROM blocks WHERE block_type='BAKED' AND state='READY' AND "
      "min_timestamp_micros <= ? AND max_timestamp_micros >= ?";
  if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    return SqliteError(db_, "failed preparing candidate blocks query");
  }
  sqlite3_bind_int64(stmt, 1, max_bound);
  sqlite3_bind_int64(stmt, 2, min_bound);

  std::vector<BlockMetadata> blocks;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    BlockMetadata m;
    m.block_id = sqlite3_column_int64(stmt, 0);
    m.block_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
    m.block_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
    m.state = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
    m.record_count = sqlite3_column_int64(stmt, 4);
    m.min_timestamp_micros = sqlite3_column_int64(stmt, 5);
    m.max_timestamp_micros = sqlite3_column_int64(stmt, 6);

    const void* agent_blob = sqlite3_column_blob(stmt, 7);
    const int agent_len = sqlite3_column_bytes(stmt, 7);
    if (agent_blob != nullptr && agent_len > 0) {
      m.agent_bloom.assign(static_cast<const char*>(agent_blob), static_cast<size_t>(agent_len));
    }

    const void* session_blob = sqlite3_column_blob(stmt, 8);
    const int session_len = sqlite3_column_bytes(stmt, 8);
    if (session_blob != nullptr && session_len > 0) {
      m.session_bloom.assign(static_cast<const char*>(session_blob),
                             static_cast<size_t>(session_len));
    }

    m.parent_block_id = sqlite3_column_int64(stmt, 9);
    blocks.push_back(std::move(m));
  }

  sqlite3_finalize(stmt);
  DLOG(INFO) << "component=metadata event=candidate_blocks count=" << blocks.size()
             << " min_bound=" << min_bound << " max_bound=" << max_bound;
  return blocks;
}

absl::StatusOr<std::vector<BlockMetadata>> MetadataStore::FindCandidateFreshBlocks(
    const QueryFilter& filter) const {
  absl::MutexLock lock(&mutex_);
  if (auto status = EnsureOpen(); !status.ok()) {
    return status;
  }

  int64_t min_bound = 0;
  int64_t max_bound = std::numeric_limits<int64_t>::max();
  if (filter.start_time.has_value()) {
    min_bound = absl::ToUnixMicros(*filter.start_time);
  }
  if (filter.end_time.has_value()) {
    max_bound = absl::ToUnixMicros(*filter.end_time);
  }

  sqlite3_stmt* stmt = nullptr;
  const char* sql =
      "SELECT block_id, block_path, block_type, state, record_count, "
      "min_timestamp_micros, max_timestamp_micros, parent_block_id "
      "FROM blocks WHERE block_type='FRESH' AND state IN ('ACTIVE', 'SEALED') AND "
      "(state='ACTIVE' OR (min_timestamp_micros <= ? AND max_timestamp_micros >= ?))";
  if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    return SqliteError(db_, "failed preparing candidate fresh blocks query");
  }
  sqlite3_bind_int64(stmt, 1, max_bound);
  sqlite3_bind_int64(stmt, 2, min_bound);

  std::vector<BlockMetadata> blocks;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    BlockMetadata m;
    m.block_id = sqlite3_column_int64(stmt, 0);
    m.block_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
    m.block_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
    m.state = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
    m.record_count = sqlite3_column_int64(stmt, 4);
    m.min_timestamp_micros = sqlite3_column_int64(stmt, 5);
    m.max_timestamp_micros = sqlite3_column_int64(stmt, 6);
    m.parent_block_id = sqlite3_column_int64(stmt, 7);
    blocks.push_back(std::move(m));
  }

  sqlite3_finalize(stmt);
  DLOG(INFO) << "component=metadata event=candidate_fresh_blocks count=" << blocks.size()
             << " min_bound=" << min_bound << " max_bound=" << max_bound;
  return blocks;
}

}  // namespace kalki
