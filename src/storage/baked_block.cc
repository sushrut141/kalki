#include "kalki/storage/baked_block.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <tuple>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "kalki/storage/bloom_filter.h"
#include "kalki/storage/compression.h"

namespace kalki {

namespace {

int64_t ToMicros(const google::protobuf::Timestamp& timestamp) {
  return static_cast<int64_t>(timestamp.seconds()) * 1000000LL +
         static_cast<int64_t>(timestamp.nanos()) / 1000LL;
}

}  // namespace

absl::Status BakedBlockWriter::Write(const std::string& path, std::vector<ProcessedRecord> records,
                                     int64_t* out_min_ts_micros, int64_t* out_max_ts_micros,
                                     std::string* out_agent_bloom, std::string* out_session_bloom) {
  if (records.empty()) {
    return absl::InvalidArgumentError("cannot bake empty block");
  }

  std::sort(records.begin(), records.end(), [](const ProcessedRecord& a, const ProcessedRecord& b) {
    const auto ta = ToMicros(a.timestamp());
    const auto tb = ToMicros(b.timestamp());
    return std::tie(ta, a.agent_id(), a.session_id()) < std::tie(tb, b.agent_id(), b.session_id());
  });

  BloomFilter agent_bloom(8192, 7);
  BloomFilter session_bloom(8192, 7);

  std::vector<std::string> compressed_records;
  compressed_records.reserve(records.size());

  BakedHeader header;
  int64_t min_ts = std::numeric_limits<int64_t>::max();
  int64_t max_ts = std::numeric_limits<int64_t>::min();

  uint64_t running_offset = 0;
  for (const auto& rec : records) {
    agent_bloom.Add(rec.agent_id());
    session_bloom.Add(rec.session_id());

    const int64_t ts = ToMicros(rec.timestamp());
    min_ts = std::min(min_ts, ts);
    max_ts = std::max(max_ts, ts);

    std::string serialized;
    if (!rec.SerializeToString(&serialized)) {
      return absl::InternalError("failed serializing processed record for baked block");
    }
    auto compressed_or = CompressZstd(serialized);
    if (!compressed_or.ok()) {
      return compressed_or.status();
    }

    BakedHeaderEntry* entry = header.add_entries();
    entry->set_timestamp_unix_micros(ts);
    entry->set_agent_id(rec.agent_id());
    entry->set_session_id(rec.session_id());
    entry->set_summary(rec.summary());
    entry->set_data_offset(running_offset);
    entry->set_data_size(static_cast<uint32_t>(compressed_or->size()));

    running_offset += static_cast<uint64_t>(compressed_or->size());
    compressed_records.push_back(std::move(*compressed_or));
  }

  header.set_min_timestamp_unix_micros(min_ts);
  header.set_max_timestamp_unix_micros(max_ts);

  std::string serialized_header;
  if (!header.SerializeToString(&serialized_header)) {
    return absl::InternalError("failed serializing baked header");
  }
  auto compressed_header_or = CompressZstd(serialized_header);
  if (!compressed_header_or.ok()) {
    return compressed_header_or.status();
  }

  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.good()) {
    return absl::InternalError(absl::StrCat("failed creating baked block: ", path));
  }

  BakedBlockFileHeader fh;
  fh.compressed_header_bytes = static_cast<uint64_t>(compressed_header_or->size());
  fh.data_section_offset = sizeof(BakedBlockFileHeader) + fh.compressed_header_bytes;

  out.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
  out.write(compressed_header_or->data(),
            static_cast<std::streamsize>(compressed_header_or->size()));
  for (const auto& chunk : compressed_records) {
    out.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
  }
  out.flush();
  if (!out.good()) {
    return absl::InternalError("failed writing baked block bytes");
  }

  if (out_min_ts_micros != nullptr) {
    *out_min_ts_micros = min_ts;
  }
  if (out_max_ts_micros != nullptr) {
    *out_max_ts_micros = max_ts;
  }
  if (out_agent_bloom != nullptr) {
    *out_agent_bloom = agent_bloom.Serialize();
  }
  if (out_session_bloom != nullptr) {
    *out_session_bloom = session_bloom.Serialize();
  }

  LOG(INFO) << "component=baked_block event=write path=" << path << " records=" << records.size();
  return absl::OkStatus();
}

absl::StatusOr<BakedHeader> BakedBlockReader::ReadHeaderOnly(const std::string& path,
                                                             uint64_t* out_data_section_offset) {
  std::ifstream in(path, std::ios::binary);
  if (!in.good()) {
    return absl::InternalError(absl::StrCat("failed opening baked block: ", path));
  }

  BakedBlockFileHeader fh;
  in.read(reinterpret_cast<char*>(&fh), sizeof(fh));
  if (!in.good()) {
    return absl::InternalError("failed reading baked block file header");
  }
  if (fh.magic != 0x314b4c42444b4142ULL) {
    return absl::InvalidArgumentError("invalid baked block magic");
  }

  std::string compressed_header(fh.compressed_header_bytes, '\0');
  in.read(compressed_header.data(), static_cast<std::streamsize>(compressed_header.size()));
  if (!in.good()) {
    return absl::InternalError("failed reading compressed header");
  }

  auto serialized_header_or = DecompressZstd(compressed_header, 8 * 1024 * 1024);
  if (!serialized_header_or.ok()) {
    return serialized_header_or.status();
  }

  BakedHeader header;
  if (!header.ParseFromString(*serialized_header_or)) {
    return absl::InternalError("failed parsing baked header proto");
  }

  if (out_data_section_offset != nullptr) {
    *out_data_section_offset = fh.data_section_offset;
  }
  return header;
}

absl::StatusOr<ProcessedRecord> BakedBlockReader::ReadRecordAt(const std::string& path,
                                                               uint64_t data_section_offset,
                                                               uint64_t offset, uint32_t size) {
  std::ifstream in(path, std::ios::binary);
  if (!in.good()) {
    return absl::InternalError(absl::StrCat("failed opening baked block: ", path));
  }

  in.seekg(static_cast<std::streamoff>(data_section_offset + offset));
  std::string compressed(size, '\0');
  in.read(compressed.data(), static_cast<std::streamsize>(compressed.size()));
  if (!in.good()) {
    return absl::InternalError("failed reading baked data record");
  }

  auto bytes_or = DecompressZstd(compressed, 32 * 1024 * 1024);
  if (!bytes_or.ok()) {
    return bytes_or.status();
  }

  ProcessedRecord record;
  if (!record.ParseFromString(*bytes_or)) {
    return absl::InternalError("failed parsing baked record");
  }
  return record;
}

}  // namespace kalki
