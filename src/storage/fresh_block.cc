#include "kalki/storage/fresh_block.h"

#include <filesystem>
#include <fstream>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"

namespace kalki {

namespace {

absl::Mutex g_single_fresh_block_mu;

}  // namespace

FreshBlockWriter::~FreshBlockWriter() { Close(); }

absl::Status FreshBlockWriter::Open(const std::string& path) {
  Close();
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  file_.open(path, std::ios::binary | std::ios::app);
  path_ = path;
  if (!file_.good()) {
    return absl::InternalError(absl::StrCat("failed to open fresh block: ", path));
  }
  return absl::OkStatus();
}

absl::Status FreshBlockWriter::Append(const ProcessedRecord& record) {
  CHECK(!path_.empty()) << "fresh block writer path is not initialized";
  absl::WriterMutexLock writer_lock(g_single_fresh_block_mu);

  if (!file_.good()) {
    return absl::FailedPreconditionError("fresh block file is not open");
  }
  std::string bytes;
  if (!record.SerializeToString(&bytes)) {
    return absl::InternalError("failed to serialize processed record");
  }
  const uint32_t len = static_cast<uint32_t>(bytes.size());
  file_.write(reinterpret_cast<const char*>(&len), sizeof(len));
  file_.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  file_.flush();
  if (!file_.good()) {
    return absl::InternalError("failed to append record to fresh block");
  }
  return absl::OkStatus();
}

void FreshBlockWriter::Close() {
  if (file_.is_open()) {
    file_.flush();
    file_.close();
  }
}

absl::StatusOr<std::vector<ProcessedRecord>> FreshBlockReader::ReadAll(const std::string& path) {
  absl::ReaderMutexLock reader_lock(g_single_fresh_block_mu);

  std::ifstream file(path, std::ios::binary);
  if (!file.good()) {
    return absl::InternalError(absl::StrCat("failed to open fresh block: ", path));
  }

  std::vector<ProcessedRecord> records;
  while (true) {
    uint32_t len = 0;
    file.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (file.eof()) {
      break;
    }
    if (!file.good()) {
      return absl::InternalError("failed reading fresh block record size");
    }

    std::string bytes(len, '\0');
    file.read(bytes.data(), len);
    if (!file.good()) {
      return absl::InternalError("failed reading fresh block record bytes");
    }

    ProcessedRecord record;
    if (!record.ParseFromString(bytes)) {
      return absl::InternalError("failed parsing fresh block record");
    }
    records.push_back(std::move(record));
  }

  return records;
}

}  // namespace kalki
