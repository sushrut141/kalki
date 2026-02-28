#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "benchmark/benchmark.h"
#include "grpcpp/grpcpp.h"
#include "kalki.grpc.pb.h"
#include "kalki/api/agent_log_service.h"
#include "kalki/common/config.h"
#include "kalki/core/database_engine.h"
#include "kalki/llm/local_embedding_client.h"

namespace {

constexpr char kExpectedAgentId[] = "agent_expected_fixed";
constexpr char kExpectedSessionId[] = "session_expected_fixed";
constexpr char kMissingAgentId[] = "agent_missing_fixed";
constexpr char kMissingSessionId[] = "session_missing_fixed";
constexpr char kExpectedQuery[] = "Did the build pass?";
constexpr char kRawConversationLog[] =
    "We started triaging a bug where the query endpoint returned stale data for some sessions. "
    "The first clue was a mismatch between candidate block counts and actual records. "
    "I pulled logs from the coordinator thread and noticed block filtering happened before "
    "timestamp validation, which looked suspicious during concurrent ingestion. "
    "We reviewed metadata snapshots and saw fresh blocks rotating while compaction queued old "
    "ones. "
    "That raised the question of lock ordering around block writers and query readers. "
    "I traced the write path from API store calls into WAL append and ingestion worker processing. "
    "The worker parsed each protobuf record, called summarization, and appended processed records "
    "to "
    "fresh blocks. During that path we updated offsets and counts, but some updates happened too "
    "often. "
    "We hypothesized this could produce transient states where query missed expected records. "
    "To test that, I wrote a reproducible script that inserted many sessions with interleaved "
    "timestamps. "
    "The script repeatedly queried by agent, by session, and by time windows while ingestion ran "
    "in parallel. "
    "In some runs we observed candidate blocks with valid ranges but missing payload reads. "
    "Next we inspected the baked header generation and confirmed sorting by timestamp, agent, and "
    "session worked. "
    "Then we examined bloom filter construction and serialization for agent and session columns. "
    "A check revealed filtering in one code path ignored bloom hints until after expensive header "
    "reads. "
    "That did not directly cause wrong answers but increased timing sensitivity under load. "
    "We optimized candidate selection to apply bloom checks in metadata selection for baked "
    "blocks. "
    "After that change the query worker scanned fewer blocks and timing behavior stabilized. "
    "Still, we found another issue in record count persistence for fresh blocks. "
    "The update statement did not scope to fresh block type, which could allow incorrect updates "
    "later. "
    "We renamed the API to make intent explicit and restricted SQL to fresh blocks only. "
    "That improved clarity and prevented accidental cross type updates. "
    "We also tightened WAL trimming semantics to avoid repeated full scans by keeping metadata "
    "counters current. "
    "The trim flow now locks WAL, rewrites a clean temporary file, swaps it, and updates offsets "
    "atomically. "
    "With that in place we re-ran stress scenarios and measured consistent throughput. "
    "On the query side, we switched similarity evaluation to embedding views to avoid copies. "
    "This reduced allocation overhead and improved thread pool efficiency at high concurrency. "
    "We added tests to ensure fresh and baked blocks are both queried correctly during "
    "transitions. "
    "Additional tests validated filters by agent, session, and timestamp combinations. "
    "For service coverage we tested store and query APIs through the gRPC service implementation "
    "directly. "
    "We intentionally used a delayed fake summarizer to simulate real model latency and verify "
    "resilience. "
    "The pipeline remained stable even with delayed summaries while write API stayed responsive. "
    "Finally we reviewed CMake options and removed optional compression branches to enforce zstd "
    "always on. "
    "This simplified runtime assumptions and made storage format deterministic across "
    "environments. "
    "We documented the state transitions for fresh sealed compacted and baked lifecycle states. "
    "We also confirmed compacted fresh files are removed after baked outputs are durable. "
    "The final validation included integration tests, lifecycle checks, and metadata filter "
    "correctness. "
    "After patching, the query endpoint consistently returned expected records for fixed filters. "
    "The bug fix was merged with benchmarks and regression coverage to prevent recurrence.";

std::string RandomId(std::mt19937* rng, const std::string& prefix) {
  std::uniform_int_distribution<int> dist(100000, 999999);
  return absl::StrCat(prefix, "_", dist(*rng));
}

double PercentileMs(std::vector<double> samples_ms, double p) {
  if (samples_ms.empty()) {
    return 0.0;
  }
  std::sort(samples_ms.begin(), samples_ms.end());
  const size_t idx =
      static_cast<size_t>(std::min<double>(samples_ms.size() - 1, (samples_ms.size() - 1) * p));
  return samples_ms[idx];
}

std::string ModelPath() {
  return std::string(KALKI_PROJECT_SOURCE_DIR) + "/third_party/models/all-minilm-v2.gguf";
}

std::string CreateTempDir(const std::string& prefix) {
  const auto dir = std::filesystem::temp_directory_path() /
                   absl::StrCat(prefix, "_", absl::ToUnixMicros(absl::Now()));
  std::filesystem::create_directories(dir);
  return dir.string();
}

kalki::DatabaseConfig BuildConfig(const std::string& base_dir) {
  kalki::DatabaseConfig cfg;
  cfg.data_dir = base_dir;
  cfg.wal_path = base_dir + "/wal.log";
  cfg.metadata_db_path = base_dir + "/metadata.db";
  cfg.fresh_block_dir = base_dir + "/blocks/fresh";
  cfg.baked_block_dir = base_dir + "/blocks/baked";
  cfg.grpc_listen_address = "127.0.0.1:0";
  cfg.embedding_model_path = ModelPath();
  cfg.embedding_threads = 2;
  cfg.max_records_per_fresh_block = 500;
  cfg.wal_trim_threshold_records = 2000;
  cfg.wal_read_batch_size = 512;
  cfg.query_worker_threads = 4;
  cfg.similarity_threshold = -1.0;
  cfg.query_timeout_ms = 5000;
  cfg.worker_timeout_ms = 500;
  cfg.ingestion_poll_interval = absl::Milliseconds(5);
  cfg.compaction_poll_interval = absl::Milliseconds(5);
  return cfg;
}

class BenchmarkEnvironment {
 public:
  explicit BenchmarkEnvironment(const std::string& prefix)
      : base_dir_(CreateTempDir(prefix)),
        config_(BuildConfig(base_dir_)),
        engine_(config_, std::make_unique<kalki::LocalEmbeddingClient>(config_.embedding_model_path,
                                                                       config_.embedding_threads)),
        service_(&engine_) {
    const auto init_status = engine_.Initialize();
    if (!init_status.ok()) {
      throw std::runtime_error(std::string(init_status.message()));
    }
  }

  ~BenchmarkEnvironment() {
    engine_.Shutdown();
    std::error_code ec;
    std::filesystem::remove_all(base_dir_, ec);
  }

  void SeedQueryableData(int count) {
    std::mt19937 rng(42);
    const absl::Time base_time = absl::Now() - absl::Minutes(10);
    for (int i = 0; i < count; ++i) {
      kalki::StoreLogRequest req;
      if (i % 25 == 0) {
        req.set_agent_id(kExpectedAgentId);
        req.set_session_id(kExpectedSessionId);
      } else {
        req.set_agent_id(RandomId(&rng, "agent"));
        req.set_session_id(RandomId(&rng, "session"));
      }
      req.set_conversation_log(kRawConversationLog);
      const int64_t micros = absl::ToUnixMicros(base_time + absl::Seconds(i));
      req.mutable_timestamp()->set_seconds(micros / 1000000LL);
      req.mutable_timestamp()->set_nanos(static_cast<int32_t>((micros % 1000000LL) * 1000));
      kalki::StoreLogResponse resp;
      grpc::ServerContext ctx;
      service_.StoreLog(&ctx, &req, &resp);
    }
    absl::SleepFor(absl::Seconds(2));
  }

  kalki::AgentLogServiceImpl* service() { return &service_; }

 private:
  std::string base_dir_;
  kalki::DatabaseConfig config_;
  kalki::DatabaseEngine engine_;
  kalki::AgentLogServiceImpl service_;
};

static void BM_AgentLogService_StoreLog(benchmark::State& state) {
  BenchmarkEnvironment env("kalki_bm_store");
  std::mt19937 rng(12345);
  std::vector<double> latencies_ms;
  latencies_ms.reserve(static_cast<size_t>(state.max_iterations));
  int64_t op_index = 0;

  const auto start_all = std::chrono::steady_clock::now();
  for (auto _ : state) {
    (void)_;
    kalki::StoreLogRequest req;
    req.set_agent_id(op_index % 100 == 0 ? kExpectedAgentId : RandomId(&rng, "agent"));
    req.set_session_id(op_index % 100 == 0 ? kExpectedSessionId : RandomId(&rng, "session"));
    req.set_conversation_log(kRawConversationLog);
    const int64_t micros = absl::ToUnixMicros(absl::Now());
    req.mutable_timestamp()->set_seconds(micros / 1000000LL);
    req.mutable_timestamp()->set_nanos(static_cast<int32_t>((micros % 1000000LL) * 1000));
    kalki::StoreLogResponse resp;
    grpc::ServerContext ctx;

    const auto op_start = std::chrono::steady_clock::now();
    const grpc::Status status = env.service()->StoreLog(&ctx, &req, &resp);
    const auto op_end = std::chrono::steady_clock::now();
    latencies_ms.push_back(std::chrono::duration<double, std::milli>(op_end - op_start).count());

    benchmark::DoNotOptimize(status.error_code());
    benchmark::DoNotOptimize(resp.status());
    benchmark::ClobberMemory();
    ++op_index;
  }
  const auto end_all = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(end_all - start_all).count();

  state.counters["qps"] = secs > 0.0 ? static_cast<double>(state.iterations()) / secs : 0.0;
  state.counters["p50_ms"] = PercentileMs(latencies_ms, 0.50);
  state.counters["p90_ms"] = PercentileMs(latencies_ms, 0.90);
  state.SetItemsProcessed(state.iterations());
}

static void BM_AgentLogService_QueryLogs(benchmark::State& state) {
  BenchmarkEnvironment env("kalki_bm_query");
  env.SeedQueryableData(10000);
  std::vector<double> latencies_ms;
  latencies_ms.reserve(static_cast<size_t>(state.max_iterations * 2));

  const auto start_all = std::chrono::steady_clock::now();
  for (auto _ : state) {
    (void)_;
    kalki::QueryRequest existing_req;
    existing_req.set_caller_agent_id("benchmark_client");
    existing_req.set_query(kExpectedQuery);
    existing_req.set_agent_id(kExpectedAgentId);
    existing_req.set_session_id(kExpectedSessionId);
    kalki::QueryResponse existing_resp;
    grpc::ServerContext existing_ctx;
    const auto existing_start = std::chrono::steady_clock::now();
    const grpc::Status existing_status =
        env.service()->QueryLogs(&existing_ctx, &existing_req, &existing_resp);
    const auto existing_end = std::chrono::steady_clock::now();
    latencies_ms.push_back(
        std::chrono::duration<double, std::milli>(existing_end - existing_start).count());
    benchmark::DoNotOptimize(existing_status.error_code());
    benchmark::DoNotOptimize(existing_resp.records_size());

    kalki::QueryRequest missing_req;
    missing_req.set_caller_agent_id("benchmark_client");
    missing_req.set_query(kExpectedQuery);
    missing_req.set_agent_id(kMissingAgentId);
    missing_req.set_session_id(kMissingSessionId);
    kalki::QueryResponse missing_resp;
    grpc::ServerContext missing_ctx;
    const auto missing_start = std::chrono::steady_clock::now();
    const grpc::Status missing_status =
        env.service()->QueryLogs(&missing_ctx, &missing_req, &missing_resp);
    const auto missing_end = std::chrono::steady_clock::now();
    latencies_ms.push_back(
        std::chrono::duration<double, std::milli>(missing_end - missing_start).count());
    benchmark::DoNotOptimize(missing_status.error_code());
    benchmark::DoNotOptimize(missing_resp.records_size());

    benchmark::ClobberMemory();
  }
  const auto end_all = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(end_all - start_all).count();
  const double total_queries = static_cast<double>(state.iterations() * 2);

  state.counters["qps"] = secs > 0.0 ? total_queries / secs : 0.0;
  state.counters["p50_ms"] = PercentileMs(latencies_ms, 0.50);
  state.counters["p90_ms"] = PercentileMs(latencies_ms, 0.90);
  state.SetItemsProcessed(static_cast<int64_t>(total_queries));
}

BENCHMARK(BM_AgentLogService_StoreLog)->Iterations(100000);
BENCHMARK(BM_AgentLogService_QueryLogs)->Iterations(10000);

}  // namespace

BENCHMARK_MAIN();
