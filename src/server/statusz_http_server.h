#pragma once

#include <atomic>
#include <string>
#include <thread>

#include "absl/status/status.h"

namespace kalki {

class AgentLogServiceImpl;

class StatuszHttpServer {
 public:
  StatuszHttpServer(std::string listen_address, const AgentLogServiceImpl* service);
  ~StatuszHttpServer();

  StatuszHttpServer(const StatuszHttpServer&) = delete;
  StatuszHttpServer& operator=(const StatuszHttpServer&) = delete;

  absl::Status Start();
  void Stop();

 private:
  absl::Status BindSocket();
  void ServeLoop();
  static std::string BuildHttpResponse(int status_code, const std::string& status_text,
                                       const std::string& content_type, const std::string& body);

  std::string listen_address_;
  const AgentLogServiceImpl* service_ = nullptr;
  int listen_fd_ = -1;
  std::atomic<bool> stop_{false};
  std::thread serve_thread_;
};

}  // namespace kalki
