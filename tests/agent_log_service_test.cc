#include <filesystem>
#include <memory>
#include <random>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "common/fake_clients.h"
#include "google/protobuf/timestamp.pb.h"
#include "grpcpp/grpcpp.h"
#include "gtest/gtest.h"
#include "kalki.grpc.pb.h"
#include "kalki/api/agent_log_service.h"
#include "kalki/common/config.h"
#include "kalki/core/database_engine.h"
#include "kalki/llm/llm_client.h"
#include "kalki/llm/local_embedding_client.h"

namespace {

constexpr char kExpectedAgentId[] = "agent_expected_fixed";
constexpr char kExpectedSessionId[] = "session_expected_fixed";
constexpr char kExpectedQuery[] = "Did the build pass?";
constexpr char kExpectedSummary[] =
    "Reviewed query behavior while ingestion was active.\n"
    "Patched block filtering and record count updates.\n"
    "Validated service responses with deterministic filters.";
constexpr char kRawConversationLog[] =
    "We began by inspecting a recurring production issue where the query endpoint appeared to miss "
    "recently ingested conversations for a specific agent session pair. The first debug pass focused "
    "on request validation and transport handling, because malformed timestamps can silently shift "
    "records outside expected windows. We confirmed request payloads looked healthy, so the team "
    "moved to pipeline internals and traced record flow from API write calls through WAL append, "
    "ingestion worker processing, and fresh block persistence. During this trace we observed that "
    "concurrent ingestion and query operations amplified timing windows around block rotation. A "
    "fresh block could be nearing seal state while query workers were building candidate sets, and "
    "that made it critical that metadata and block files stayed consistent at each transition point. "
    "We verified lock boundaries around the active fresh writer and ensured queries used read locks "
    "when scanning fresh blocks. The investigation then turned to metadata candidate pruning, where "
    "we used bloom filters for agent and session identifiers. Earlier logic deferred some bloom-based "
    "skips until later stages, creating unnecessary scans. We tightened this by filtering candidate "
    "baked blocks as early as metadata selection. Next we reviewed WAL trimming behavior because large "
    "append bursts could grow files quickly and obscure regressions. The corrected trim flow uses a "
    "maintenance lock, rewrites a clean temporary WAL, atomically swaps files, and updates both WAL "
    "offset and count in metadata. We also validated that fresh block record counts are only updated "
    "for fresh blocks, preventing accidental cross-type updates. After applying fixes, stress runs "
    "showed stable service write acknowledgements and consistent query results for filtered requests. "
    "The final verification included API-level tests through AgentLogService and confirmed the output "
    "payload matched expected conversation records while irrelevant random records were excluded.";

std::string CreateTempDir() {
  const auto dir = std::filesystem::temp_directory_path() /
                   absl::StrCat("kalki_agent_service_test_", absl::ToUnixMicros(absl::Now()));
  std::filesystem::create_directories(dir);
  return dir.string();
}

std::string ModelPath() {
  return std::string(KALKI_PROJECT_SOURCE_DIR) + "/third_party/models/all-minilm-v2.gguf";
}

std::string RandomId(std::mt19937* rng, const std::string& prefix) {
  std::uniform_int_distribution<int> dist(100000, 999999);
  return absl::StrCat(prefix, "_", dist(*rng));
}

kalki::DatabaseConfig BuildConfig(const std::string& base_dir) {
  kalki::DatabaseConfig cfg;
  cfg.data_dir = base_dir;
  cfg.wal_path = base_dir + "/wal.log";
  cfg.metadata_db_path = base_dir + "/metadata.db";
  cfg.fresh_block_dir = base_dir + "/blocks/fresh";
  cfg.baked_block_dir = base_dir + "/blocks/baked";
  cfg.grpc_listen_address = "127.0.0.1:0";
  cfg.llm_api_key = "unused";
  cfg.llm_model = "unused";
  cfg.embedding_model_path = ModelPath();
  cfg.embedding_threads = 2;
  cfg.max_records_per_fresh_block = 50;
  cfg.wal_trim_threshold_records = 200;
  cfg.wal_read_batch_size = 256;
  cfg.query_worker_threads = 2;
  cfg.similarity_threshold = -1.0;
  cfg.query_timeout_ms = 5000;
  cfg.worker_timeout_ms = 500;
  cfg.ingestion_poll_interval = absl::Milliseconds(5);
  cfg.compaction_poll_interval = absl::Milliseconds(5);
  return cfg;
}

class DelayedFakeLlmClient final : public kalki::LlmClient {
 public:
  absl::StatusOr<std::vector<std::string>> SummarizeConversation(
      const std::string& conversation_log) override {
    (void)conversation_log;
    absl::SleepFor(absl::Milliseconds(200));
    return std::vector<std::string>{kExpectedSummary};
  }
};

google::protobuf::Timestamp ToTimestamp(absl::Time t) {
  const int64_t micros = absl::ToUnixMicros(t);
  google::protobuf::Timestamp ts;
  ts.set_seconds(micros / 1000000LL);
  ts.set_nanos(static_cast<int32_t>((micros % 1000000LL) * 1000));
  return ts;
}

TEST(AgentLogServiceTest, StoreLog) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<DelayedFakeLlmClient>(),
                               std::make_unique<kalki::LocalEmbeddingClient>(
                                   config.embedding_model_path, config.embedding_threads));
  const auto init_status = engine.Initialize();
  kalki::AgentLogServiceImpl service(&engine);

  EXPECT_TRUE(init_status.ok());
  EXPECT_TRUE(std::filesystem::exists(config.wal_path));

  kalki::StoreLogRequest req;
  req.set_agent_id(kExpectedAgentId);
  req.set_session_id(kExpectedSessionId);
  req.set_conversation_log(kRawConversationLog);
  *req.mutable_timestamp() = ToTimestamp(absl::Now());
  kalki::StoreLogResponse resp;
  grpc::ServerContext ctx;
  const grpc::Status status = service.StoreLog(&ctx, &req, &resp);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(resp.status(), kalki::StoreLogResponse::STATUS_OK);
  EXPECT_TRUE(resp.error_message().empty());

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

TEST(AgentLogServiceTest, QueryLogs) {
  const std::string base_dir = CreateTempDir();
  const kalki::DatabaseConfig config = BuildConfig(base_dir);
  kalki::DatabaseEngine engine(config, std::make_unique<DelayedFakeLlmClient>(),
                               std::make_unique<kalki::LocalEmbeddingClient>(
                                   config.embedding_model_path, config.embedding_threads));
  const auto init_status = engine.Initialize();
  kalki::AgentLogServiceImpl service(&engine);
  std::mt19937 rng(123);

  EXPECT_TRUE(init_status.ok());
  EXPECT_TRUE(std::filesystem::exists(config.baked_block_dir));

  for (int i = 0; i < 30; ++i) {
    kalki::StoreLogRequest req;
    if (i < 3) {
      req.set_agent_id(kExpectedAgentId);
      req.set_session_id(kExpectedSessionId);
    } else {
      req.set_agent_id(RandomId(&rng, "agent"));
      req.set_session_id(RandomId(&rng, "session"));
    }
    req.set_conversation_log(kRawConversationLog);
    *req.mutable_timestamp() = ToTimestamp(absl::Now() - absl::Seconds(30 - i));
    kalki::StoreLogResponse resp;
    grpc::ServerContext ctx;
    const grpc::Status status = service.StoreLog(&ctx, &req, &resp);
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(resp.status(), kalki::StoreLogResponse::STATUS_OK);
  }
  absl::SleepFor(absl::Seconds(2));

  kalki::QueryRequest query_req;
  query_req.set_caller_agent_id("tester");
  query_req.set_query(kExpectedQuery);
  query_req.set_agent_id(kExpectedAgentId);
  query_req.set_session_id(kExpectedSessionId);
  kalki::QueryResponse query_resp;
  grpc::ServerContext query_ctx;
  const grpc::Status query_status = service.QueryLogs(&query_ctx, &query_req, &query_resp);

  EXPECT_TRUE(query_status.ok());
  EXPECT_EQ(query_resp.status(), kalki::QueryResponse::STATUS_COMPLETE);
  EXPECT_EQ(query_resp.records_size(), 3);
  for (const auto& record : query_resp.records()) {
    EXPECT_EQ(record.agent_id(), kExpectedAgentId);
    EXPECT_EQ(record.session_id(), kExpectedSessionId);
    EXPECT_EQ(record.raw_conversation_log(), kRawConversationLog);
  }

  engine.Shutdown();
  std::error_code ec;
  const auto removed = std::filesystem::remove_all(base_dir, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(removed, 0);
  EXPECT_FALSE(std::filesystem::exists(base_dir));
}

}  // namespace
