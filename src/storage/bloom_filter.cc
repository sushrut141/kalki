#include "kalki/storage/bloom_filter.h"

#include <cstring>
#include <string>

#include "absl/hash/hash.h"

namespace kalki {

BloomFilter::BloomFilter(size_t bit_count, size_t hash_count)
    : bit_count_(bit_count), hash_count_(hash_count), bits_((bit_count + 7) / 8, 0) {}

uint64_t BloomFilter::HashAt(const std::string& value, size_t index) const {
  return absl::HashOf(value, index, bit_count_, hash_count_);
}

void BloomFilter::Add(const std::string& value) {
  for (size_t i = 0; i < hash_count_; ++i) {
    const uint64_t hash = HashAt(value, i);
    const size_t bit = hash % bit_count_;
    bits_[bit / 8] |= static_cast<uint8_t>(1u << (bit % 8));
  }
}

bool BloomFilter::PossiblyContains(const std::string& value) const {
  for (size_t i = 0; i < hash_count_; ++i) {
    const uint64_t hash = HashAt(value, i);
    const size_t bit = hash % bit_count_;
    if ((bits_[bit / 8] & static_cast<uint8_t>(1u << (bit % 8))) == 0) {
      return false;
    }
  }
  return true;
}

std::string BloomFilter::Serialize() const {
  std::string out;
  out.resize(sizeof(uint64_t) * 2 + bits_.size());
  uint64_t* header = reinterpret_cast<uint64_t*>(out.data());
  header[0] = static_cast<uint64_t>(bit_count_);
  header[1] = static_cast<uint64_t>(hash_count_);
  std::memcpy(out.data() + sizeof(uint64_t) * 2, bits_.data(), bits_.size());
  return out;
}

BloomFilter BloomFilter::Deserialize(const std::string& bytes) {
  if (bytes.size() < sizeof(uint64_t) * 2) {
    return BloomFilter(1024, 3);
  }
  const uint64_t* header = reinterpret_cast<const uint64_t*>(bytes.data());
  BloomFilter filter(static_cast<size_t>(header[0]), static_cast<size_t>(header[1]));
  const size_t data_size = bytes.size() - sizeof(uint64_t) * 2;
  filter.bits_.resize(data_size);
  std::memcpy(filter.bits_.data(), bytes.data() + sizeof(uint64_t) * 2, data_size);
  return filter;
}

}  // namespace kalki
