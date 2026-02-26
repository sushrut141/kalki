#include "kalki/common/thread_pool.h"

#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"

namespace kalki {

ThreadPool::ThreadPool(int thread_count) : thread_count_(thread_count) {}

ThreadPool::~ThreadPool() { Stop(); }

absl::Status ThreadPool::Start() {
  absl::MutexLock lock(&mutex_);
  if (started_) {
    return absl::OkStatus();
  }
  started_ = true;
  stop_ = false;
  threads_.reserve(thread_count_);
  for (int i = 0; i < thread_count_; ++i) {
    threads_.emplace_back([this]() { RunWorker(); });
  }
  LOG(INFO) << "thread_pool started threads=" << thread_count_;
  return absl::OkStatus();
}

void ThreadPool::Stop() {
  std::vector<std::thread> threads_to_join;
  {
    absl::MutexLock lock(&mutex_);
    if (!started_) {
      return;
    }
    stop_ = true;
    threads_to_join = std::move(threads_);
  }

  for (auto& t : threads_to_join) {
    if (t.joinable()) {
      t.join();
    }
  }

  absl::MutexLock lock(&mutex_);
  started_ = false;
  queue_.clear();
}

absl::Status ThreadPool::Submit(std::function<void()> task) {
  absl::MutexLock lock(&mutex_);
  if (!started_) {
    return absl::FailedPreconditionError("thread pool not started");
  }
  queue_.push_back(std::move(task));
  DLOG(INFO) << "component=thread_pool event=task_enqueued queue_size=" << queue_.size();
  return absl::OkStatus();
}

void ThreadPool::RunWorker() {
  while (true) {
    std::function<void()> task;
    {
      absl::MutexLock lock(&mutex_);
      mutex_.Await(absl::Condition(this, &ThreadPool::HasWork));
      if (stop_ && queue_.empty()) {
        break;
      }
      task = std::move(queue_.front());
      queue_.erase(queue_.begin());
    }
    if (task) {
      task();
    }
  }
}

bool ThreadPool::HasWork() const { return stop_ || !queue_.empty(); }

}  // namespace kalki
