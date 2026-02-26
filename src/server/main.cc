#include <memory>
#include <string>

#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "grpcpp/grpcpp.h"
#include "kalki/api/agent_log_service.h"
#include "kalki/common/config.h"
#include "kalki/common/logging.h"
#include "kalki/core/database_engine.h"

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  kalki::InitializeLogging();

  kalki::DatabaseConfig config = kalki::LoadConfigFromFlags();
  kalki::DatabaseEngine engine(config);

  auto init_status = engine.Initialize();
  if (!init_status.ok()) {
    LOG(ERROR) << "component=server event=init_failed status=" << init_status;
    return 1;
  }

  kalki::AgentLogServiceImpl service(&engine);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(config.grpc_listen_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (!server) {
    LOG(ERROR) << "component=server event=grpc_start_failed addr=" << config.grpc_listen_address;
    return 1;
  }

  LOG(INFO) << "component=server event=started addr=" << config.grpc_listen_address;
  server->Wait();

  engine.Shutdown();
  return 0;
}
