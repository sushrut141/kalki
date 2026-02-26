#pragma once

#include <functional>
#include <thread>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

namespace kalki {

// Thread-safe task execution pool.
class ThreadPool {
 public:
  explicit ThreadPool(int thread_count);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  absl::Status Start() ABSL_LOCKS_EXCLUDED(mutex_);
  void Stop() ABSL_LOCKS_EXCLUDED(mutex_);
  absl::Status Submit(std::function<void()> task) ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  bool HasWork() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void RunWorker();

  int thread_count_;
  bool started_ ABSL_GUARDED_BY(mutex_) = false;
  bool stop_ ABSL_GUARDED_BY(mutex_) = false;

  absl::Mutex mutex_;
  absl::InlinedVector<std::function<void()>, 16> queue_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::thread> threads_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace kalki
