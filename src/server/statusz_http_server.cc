#include "statusz_http_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "kalki/api/agent_log_service.h"

namespace kalki {

namespace {

absl::Status ErrnoStatus(const std::string& prefix) {
  return absl::InternalError(absl::StrCat(prefix, ": ", std::strerror(errno)));
}

bool ParseListenAddress(absl::string_view listen_address, std::string* host, int* port) {
  const size_t pos = listen_address.rfind(':');
  if (pos == absl::string_view::npos || pos == 0 || pos + 1 >= listen_address.size()) {
    return false;
  }
  *host = std::string(listen_address.substr(0, pos));
  try {
    *port = std::stoi(std::string(listen_address.substr(pos + 1)));
  } catch (...) {
    return false;
  }
  return *port > 0 && *port <= 65535;
}

}  // namespace

StatuszHttpServer::StatuszHttpServer(std::string listen_address, const AgentLogServiceImpl* service)
    : listen_address_(std::move(listen_address)), service_(service) {}

StatuszHttpServer::~StatuszHttpServer() { Stop(); }

absl::Status StatuszHttpServer::Start() {
  if (service_ == nullptr) {
    return absl::InvalidArgumentError("statusz service is null");
  }
  auto status = BindSocket();
  if (!status.ok()) {
    return status;
  }
  serve_thread_ = std::thread([this]() { ServeLoop(); });
  LOG(INFO) << "component=statusz event=started addr=" << listen_address_;
  return absl::OkStatus();
}

void StatuszHttpServer::Stop() {
  if (stop_.exchange(true, std::memory_order_relaxed)) {
    return;
  }
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  if (serve_thread_.joinable()) {
    serve_thread_.join();
  }
}

absl::Status StatuszHttpServer::BindSocket() {
  std::string host;
  int port = 0;
  if (!ParseListenAddress(listen_address_, &host, &port)) {
    return absl::InvalidArgumentError(
        absl::StrCat("invalid statusz listen address: ", listen_address_));
  }

  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    return ErrnoStatus("statusz socket() failed");
  }

  int yes = 1;
  if (::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) != 0) {
    const absl::Status status = ErrnoStatus("statusz setsockopt() failed");
    ::close(listen_fd_);
    listen_fd_ = -1;
    return status;
  }

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (host == "0.0.0.0") {
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  } else {
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
      ::close(listen_fd_);
      listen_fd_ = -1;
      return absl::InvalidArgumentError(
          absl::StrCat("statusz only supports IPv4 host, got: ", host));
    }
  }

  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    const absl::Status status = ErrnoStatus("statusz bind() failed");
    ::close(listen_fd_);
    listen_fd_ = -1;
    return status;
  }
  if (::listen(listen_fd_, 64) != 0) {
    const absl::Status status = ErrnoStatus("statusz listen() failed");
    ::close(listen_fd_);
    listen_fd_ = -1;
    return status;
  }
  return absl::OkStatus();
}

void StatuszHttpServer::ServeLoop() {
  while (!stop_.load(std::memory_order_relaxed)) {
    const int conn_fd = ::accept(listen_fd_, nullptr, nullptr);
    if (conn_fd < 0) {
      if (stop_.load(std::memory_order_relaxed)) {
        break;
      }
      continue;
    }

    char buffer[4096];
    const ssize_t read_bytes = ::recv(conn_fd, buffer, sizeof(buffer), 0);
    if (read_bytes <= 0) {
      ::close(conn_fd);
      continue;
    }

    std::string response;
    const std::string request(buffer, static_cast<size_t>(read_bytes));
    if (request.rfind("GET /statusz ", 0) == 0 || request.rfind("GET /statusz?", 0) == 0) {
      auto html_or = service_->RenderStatuszHtml();
      if (html_or.ok()) {
        response = BuildHttpResponse(200, "OK", "text/html; charset=utf-8", *html_or);
      } else {
        response = BuildHttpResponse(500, "Internal Server Error", "text/plain; charset=utf-8",
                                     absl::StrCat("statusz failed: ", html_or.status()));
      }
    } else {
      response = BuildHttpResponse(404, "Not Found", "text/plain; charset=utf-8", "not found");
    }

    const size_t total = response.size();
    size_t written = 0;
    while (written < total) {
      const ssize_t n = ::send(conn_fd, response.data() + written, total - written, 0);
      if (n <= 0) {
        break;
      }
      written += static_cast<size_t>(n);
    }
    ::close(conn_fd);
  }
}

std::string StatuszHttpServer::BuildHttpResponse(int status_code, const std::string& status_text,
                                                 const std::string& content_type,
                                                 const std::string& body) {
  return absl::StrCat("HTTP/1.1 ", status_code, " ", status_text, "\r\n",
                      "Content-Type: ", content_type, "\r\n", "Content-Length: ", body.size(),
                      "\r\n", "Connection: close\r\n\r\n", body);
}

}  // namespace kalki
