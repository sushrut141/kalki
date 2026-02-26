#pragma once

#include <algorithm>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace kalki {

struct RetryOptions {
  int max_attempts = 3;
  absl::Duration initial_backoff = absl::Milliseconds(100);
  double backoff_multiplier = 2.0;
  absl::Duration max_backoff = absl::Seconds(2);
};

template <typename Fn, typename ShouldRetry>
auto RetryStatusOr(Fn&& fn, ShouldRetry&& should_retry, const RetryOptions& options)
    -> decltype(fn()) {
  using ReturnT = decltype(fn());

  ReturnT last = fn();
  if (options.max_attempts <= 1 || last.ok()) {
    return last;
  }

  absl::Duration backoff = options.initial_backoff;
  for (int attempt = 2; attempt <= options.max_attempts; ++attempt) {
    if (!should_retry(last.status())) {
      return last;
    }

    absl::SleepFor(backoff);
    backoff = std::min(options.max_backoff, backoff * options.backoff_multiplier);

    last = fn();
    if (last.ok()) {
      return last;
    }
  }

  return last;
}

}  // namespace kalki
