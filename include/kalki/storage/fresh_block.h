#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "kalki.pb.h"

namespace kalki {

class FreshBlockWriter {
 public:
  FreshBlockWriter() = default;
  ~FreshBlockWriter();

  absl::Status Open(const std::string& path);
  absl::Status Append(const ProcessedRecord& record);
  void Close();

 private:
  std::ofstream file_;
  std::string path_;
};

class FreshBlockReader {
 public:
  static absl::StatusOr<std::vector<ProcessedRecord>> ReadAll(const std::string& path);
};

}  // namespace kalki
