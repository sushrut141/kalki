#pragma once

#include <memory>

#include "grpcpp/grpcpp.h"
#include "kalki.grpc.pb.h"
#include "kalki/core/database_engine.h"

namespace kalki {

class AgentLogServiceImpl final : public AgentLogService::Service {
 public:
  explicit AgentLogServiceImpl(DatabaseEngine* engine);

  grpc::Status StoreLog(grpc::ServerContext* context, const StoreLogRequest* request,
                        StoreLogResponse* response) override;

  grpc::Status QueryLogs(grpc::ServerContext* context, const QueryRequest* request,
                         QueryResponse* response) override;

 private:
  DatabaseEngine* engine_;
};

}  // namespace kalki
