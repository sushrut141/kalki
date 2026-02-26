#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "kalki.pb.h"

namespace kalki {

struct BakedBlockFileHeader {
  uint64_t magic = 0x314b4c42444b4142ULL;  // BAKDBLK1
  uint32_t version = 1;
  uint32_t reserved = 0;
  uint64_t compressed_header_bytes = 0;
  uint64_t data_section_offset = 0;
};

class BakedBlockWriter {
 public:
  static absl::Status Write(const std::string& path, std::vector<ProcessedRecord> records,
                            int64_t* out_min_ts_micros, int64_t* out_max_ts_micros,
                            std::string* out_agent_bloom, std::string* out_session_bloom);
};

class BakedBlockReader {
 public:
  static absl::StatusOr<BakedHeader> ReadHeaderOnly(const std::string& path,
                                                    uint64_t* out_data_section_offset);
  static absl::StatusOr<ProcessedRecord> ReadRecordAt(const std::string& path,
                                                      uint64_t data_section_offset, uint64_t offset,
                                                      uint32_t size);
};

}  // namespace kalki
