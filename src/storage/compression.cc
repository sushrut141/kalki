#include "kalki/storage/compression.h"

#include <algorithm>
#include <string>

#include "absl/status/status.h"
#include <zstd.h>

namespace kalki {

absl::StatusOr<std::string> CompressZstd(const std::string& input) {
  const size_t bound = ZSTD_compressBound(input.size());
  std::string output(bound, '\0');
  const size_t compressed_size =
      ZSTD_compress(output.data(), output.size(), input.data(), input.size(), 3);
  if (ZSTD_isError(compressed_size)) {
    return absl::InternalError("zstd compression failed");
  }
  output.resize(compressed_size);
  return output;
}

absl::StatusOr<std::string> DecompressZstd(const std::string& input,
                                           size_t expected_max_output_size) {
  unsigned long long frame_size = ZSTD_getFrameContentSize(input.data(), input.size());
  if (frame_size == ZSTD_CONTENTSIZE_ERROR) {
    return absl::InvalidArgumentError("invalid zstd frame");
  }
  if (frame_size == ZSTD_CONTENTSIZE_UNKNOWN) {
    frame_size = std::max<size_t>(expected_max_output_size, 1);
  }
  std::string output(static_cast<size_t>(frame_size), '\0');
  const size_t decompressed =
      ZSTD_decompress(output.data(), output.size(), input.data(), input.size());
  if (ZSTD_isError(decompressed)) {
    return absl::InternalError("zstd decompression failed");
  }
  output.resize(decompressed);
  return output;
}

}  // namespace kalki
