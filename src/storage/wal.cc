#include "kalki/storage/wal.h"

#include <filesystem>
#include <fstream>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace kalki {

namespace {

constexpr size_t kLengthPrefixBytes = sizeof(uint32_t);

}  // namespace

WalStore::WalStore(std::string wal_path) : wal_path_(std::move(wal_path)) {}

absl::Status WalStore::Initialize() {
  std::filesystem::path path(wal_path_);
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(wal_path_, std::ios::binary | std::ios::app);
  if (!file.good()) {
    return absl::InternalError(absl::StrCat("failed to open WAL: ", wal_path_));
  }
  LOG(INFO) << "component=wal event=initialized path=" << wal_path_;
  return absl::OkStatus();
}

absl::StatusOr<int64_t> WalStore::Append(const std::string& payload_bytes) {
  absl::MutexLock lock(&mutex_);
  std::fstream file(wal_path_, std::ios::binary | std::ios::in | std::ios::out);
  if (!file.good()) {
    return absl::InternalError("failed to open WAL for append");
  }

  file.seekp(0, std::ios::end);
  const int64_t start_offset = static_cast<int64_t>(file.tellp());

  const uint32_t len = static_cast<uint32_t>(payload_bytes.size());
  file.write(reinterpret_cast<const char*>(&len), sizeof(len));
  file.write(payload_bytes.data(), static_cast<std::streamsize>(payload_bytes.size()));
  file.flush();

  if (!file.good()) {
    return absl::InternalError("failed writing WAL payload");
  }

  LOG(INFO) << "component=wal event=append offset=" << start_offset
            << " bytes=" << payload_bytes.size();
  DLOG(INFO) << "component=wal event=append_debug path=" << wal_path_
             << " start_offset=" << start_offset;
  return start_offset;
}

absl::StatusOr<std::vector<WalEnvelope>> WalStore::ReadBatchFromOffset(int64_t offset,
                                                                       int max_records) {
  absl::MutexLock lock(&mutex_);
  std::ifstream file(wal_path_, std::ios::binary);
  if (!file.good()) {
    return absl::InternalError("failed to open WAL for read");
  }

  file.seekg(offset);
  std::vector<WalEnvelope> out;
  out.reserve(static_cast<size_t>(max_records));

  for (int i = 0; i < max_records; ++i) {
    const int64_t start = static_cast<int64_t>(file.tellg());
    uint32_t len = 0;
    file.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (file.eof()) {
      break;
    }
    if (!file.good()) {
      return absl::InternalError("failed reading WAL record length");
    }

    std::string payload(len, '\0');
    file.read(payload.data(), len);
    if (!file.good()) {
      return absl::InternalError("failed reading WAL record payload");
    }

    const int64_t next = static_cast<int64_t>(file.tellg());
    out.push_back(
        WalEnvelope{.start_offset = start, .next_offset = next, .payload = std::move(payload)});
  }

  DLOG(INFO) << "component=wal event=read_batch offset=" << offset << " records=" << out.size();
  return out;
}

absl::Status WalStore::LockForMaintenance() {
  mutex_.Lock();
  DLOG(INFO) << "component=wal event=maintenance_lock_acquired";
  return absl::OkStatus();
}

void WalStore::UnlockForMaintenance() {
  DLOG(INFO) << "component=wal event=maintenance_lock_released";
  mutex_.Unlock();
}

absl::StatusOr<WalStore::TrimResult> WalStore::TrimToLastRecordsLocked(int keep_records,
                                                                       int64_t current_offset) {
  TrimResult result;
  result.new_offset = current_offset;

  if (keep_records <= 0) {
    return absl::InvalidArgumentError("keep_records must be positive");
  }

  std::ifstream file(wal_path_, std::ios::binary);
  if (!file.good()) {
    return absl::InternalError("failed to open WAL for trim scan");
  }

  std::vector<int64_t> record_starts;
  std::vector<int64_t> record_nexts;
  while (true) {
    const int64_t start = static_cast<int64_t>(file.tellg());
    uint32_t len = 0;
    file.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (file.eof()) {
      break;
    }
    if (!file.good()) {
      return absl::InternalError("failed reading WAL length during trim scan");
    }

    file.seekg(static_cast<std::streamoff>(len), std::ios::cur);
    if (!file.good()) {
      return absl::InternalError("failed seeking WAL payload during trim scan");
    }
    const int64_t next = static_cast<int64_t>(file.tellg());
    record_starts.push_back(start);
    record_nexts.push_back(next);
  }

  result.total_records = static_cast<int64_t>(record_starts.size());
  if (result.total_records <= keep_records) {
    result.kept_records = result.total_records;
    return result;
  }

  const int64_t keep_count = std::min<int64_t>(keep_records, result.total_records);
  const int64_t keep_start_idx = result.total_records - keep_count;
  const int64_t keep_start_offset = record_starts[static_cast<size_t>(keep_start_idx)];
  const int64_t old_eof = record_nexts.back();

  file.clear();
  file.seekg(keep_start_offset, std::ios::beg);
  if (!file.good()) {
    return absl::InternalError("failed seeking WAL keep-start offset");
  }

  const std::string temp_path = wal_path_ + ".trim_tmp";
  std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
  if (!out.good()) {
    return absl::InternalError("failed opening WAL trim temp file");
  }

  int64_t kept = 0;
  for (int64_t i = keep_start_idx; i < result.total_records; ++i) {
    uint32_t len = 0;
    file.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!file.good()) {
      out.close();
      std::filesystem::remove(temp_path);
      return absl::InternalError("failed reading WAL record length during trim copy");
    }

    std::string payload(len, '\0');
    file.read(payload.data(), static_cast<std::streamsize>(len));
    if (!file.good()) {
      out.close();
      std::filesystem::remove(temp_path);
      return absl::InternalError("failed reading WAL record payload during trim copy");
    }

    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!out.good()) {
      out.close();
      std::filesystem::remove(temp_path);
      return absl::InternalError("failed writing WAL record during trim copy");
    }
    ++kept;
  }

  out.flush();
  out.close();
  file.close();

  if (current_offset <= keep_start_offset) {
    result.new_offset = 0;
  } else if (current_offset >= old_eof) {
    result.new_offset = record_nexts.back() - keep_start_offset;
  } else {
    result.new_offset = current_offset - keep_start_offset;
  }
  if (result.new_offset < 0) {
    result.new_offset = 0;
  }

  std::error_code ec;
  std::filesystem::rename(temp_path, wal_path_, ec);
  if (ec) {
    std::filesystem::remove(temp_path);
    return absl::InternalError(absl::StrCat("failed replacing WAL during trim: ", ec.message()));
  }

  result.trimmed = true;
  result.kept_records = kept;
  LOG(INFO) << "component=wal event=trimmed total_records=" << result.total_records
            << " kept_records=" << result.kept_records << " new_offset=" << result.new_offset;
  return result;
}

}  // namespace kalki
