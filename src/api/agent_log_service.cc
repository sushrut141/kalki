#include "kalki/api/agent_log_service.h"

#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"

namespace kalki {

namespace {

absl::Time FromProtoTimestamp(const google::protobuf::Timestamp& ts) {
  return absl::FromUnixSeconds(ts.seconds()) + absl::Nanoseconds(ts.nanos());
}

google::protobuf::Timestamp ToProtoTimestamp(absl::Time t) {
  const int64_t micros = absl::ToUnixMicros(t);
  google::protobuf::Timestamp ts;
  ts.set_seconds(micros / 1000000LL);
  ts.set_nanos(static_cast<int32_t>((micros % 1000000LL) * 1000));
  return ts;
}

}  // namespace

AgentLogServiceImpl::AgentLogServiceImpl(DatabaseEngine* engine) : engine_(engine) {}

grpc::Status AgentLogServiceImpl::StoreLog(grpc::ServerContext* context,
                                           const StoreLogRequest* request,
                                           StoreLogResponse* response) {
  (void)context;
  DLOG(INFO) << "component=api event=store_request agent_id=" << request->agent_id()
             << " session_id=" << request->session_id();
  if (request->agent_id().empty() || request->session_id().empty() ||
      request->conversation_log().empty()) {
    response->set_status(StoreLogResponse::STATUS_INVALID_ARGUMENT);
    response->set_error_message("agent_id, session_id and conversation_log are required");
    return grpc::Status::OK;
  }

  absl::Time timestamp = absl::Now();
  if (request->timestamp().seconds() != 0 || request->timestamp().nanos() != 0) {
    timestamp = FromProtoTimestamp(request->timestamp());
  }

  auto st = engine_->AppendConversation(request->agent_id(), request->session_id(),
                                        request->conversation_log(), timestamp);
  if (!st.ok()) {
    response->set_status(StoreLogResponse::STATUS_INTERNAL_ERROR);
    response->set_error_message(std::string(st.message()));
    LOG(ERROR) << "component=api event=store_failed status=" << st;
    return grpc::Status::OK;
  }

  response->set_status(StoreLogResponse::STATUS_OK);
  return grpc::Status::OK;
}

grpc::Status AgentLogServiceImpl::QueryLogs(grpc::ServerContext* context,
                                            const QueryRequest* request, QueryResponse* response) {
  (void)context;
  DLOG(INFO) << "component=api event=query_request caller_agent_id=" << request->caller_agent_id();
  if (request->caller_agent_id().empty() || request->query().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "caller_agent_id and query are required");
  }

  QueryFilter filter;
  filter.caller_agent_id = request->caller_agent_id();
  if (request->has_agent_id()) {
    filter.agent_id = request->agent_id();
  }
  if (request->has_session_id()) {
    filter.session_id = request->session_id();
  }
  if (request->has_start_time()) {
    filter.start_time = FromProtoTimestamp(request->start_time());
  }
  if (request->has_end_time()) {
    filter.end_time = FromProtoTimestamp(request->end_time());
  }

  auto result_or = engine_->QueryLogs(request->query(), filter);
  if (!result_or.ok()) {
    response->set_status(QueryResponse::STATUS_INTERNAL_ERROR);
    response->set_error_message(std::string(result_or.status().message()));
    return grpc::Status::OK;
  }

  const QueryExecutionResult& result = *result_or;
  switch (result.status) {
    case QueryCompletionStatus::kComplete:
      response->set_status(QueryResponse::STATUS_COMPLETE);
      break;
    case QueryCompletionStatus::kIncomplete:
      response->set_status(QueryResponse::STATUS_INCOMPLETE);
      break;
    case QueryCompletionStatus::kInternalError:
      response->set_status(QueryResponse::STATUS_INTERNAL_ERROR);
      break;
  }

  for (const auto& record : result.records) {
    AgentLogResult* out = response->add_records();
    out->set_agent_id(record.agent_id);
    out->set_session_id(record.session_id);
    *out->mutable_timestamp() = ToProtoTimestamp(record.timestamp);
    out->set_raw_conversation_log(record.raw_conversation_log);
  }
  response->set_error_message(result.error_message);

  LOG(INFO) << "component=api event=query_done status=" << static_cast<int>(result.status)
            << " records=" << result.records.size();

  return grpc::Status::OK;
}

}  // namespace kalki
