#pragma once

#include <optional>

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace kalki {

template <typename T>
class TaskQueue {
 public:
  void Push(T task) {
    absl::MutexLock lock(&mutex_);
    queue_.push_back(std::move(task));
  }

  std::optional<T> PopWithTimeout(absl::Duration timeout) {
    const absl::Time deadline = absl::Now() + timeout;
    while (absl::Now() < deadline) {
      {
        absl::MutexLock lock(&mutex_);
        if (!queue_.empty()) {
          T item = std::move(queue_.front());
          queue_.erase(queue_.begin());
          return item;
        }
      }
      absl::SleepFor(absl::Milliseconds(1));
    }
    return std::nullopt;
  }

 private:
  absl::Mutex mutex_;
  absl::InlinedVector<T, 32> queue_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace kalki
