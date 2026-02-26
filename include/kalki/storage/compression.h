#pragma once

#include <string>

#include "absl/status/statusor.h"

namespace kalki {

absl::StatusOr<std::string> CompressZstd(const std::string& input);
absl::StatusOr<std::string> DecompressZstd(const std::string& input,
                                           size_t expected_max_output_size);

}  // namespace kalki
