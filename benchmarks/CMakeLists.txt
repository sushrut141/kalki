find_package(benchmark CONFIG QUIET)
if (NOT benchmark_FOUND)
  find_package(benchmark REQUIRED)
endif ()

add_executable(kalki_agent_log_service_benchmark
  agent_log_service_benchmark.cc
)

target_compile_definitions(kalki_agent_log_service_benchmark
  PRIVATE KALKI_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries(kalki_agent_log_service_benchmark PRIVATE
  kalki::api
  kalki::core
  kalki::llm
  benchmark::benchmark
  gRPC::grpc++
  absl::strings
  absl::time
)
