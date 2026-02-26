add_library(kalki_api SHARED
  agent_log_service.cc
)
add_library(kalki::api ALIAS kalki_api)

target_include_directories(kalki_api
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(kalki_api
  PUBLIC
    kalki::common
    kalki::core
    kalki::proto
    gRPC::grpc++
    absl::log
    absl::status
    absl::strings
    absl::time
)
