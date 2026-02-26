#include "kalki/storage/wal.h"

#include <filesystem>
#include <fstream>

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

}  // namespace kalki
