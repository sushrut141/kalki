#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kalki {

class BloomFilter {
 public:
  BloomFilter(size_t bit_count, size_t hash_count);

  void Add(const std::string& value);
  bool PossiblyContains(const std::string& value) const;

  std::string Serialize() const;
  static BloomFilter Deserialize(const std::string& bytes);

 private:
  uint64_t HashAt(const std::string& value, size_t index) const;

  size_t bit_count_;
  size_t hash_count_;
  std::vector<uint8_t> bits_;
};

}  // namespace kalki
