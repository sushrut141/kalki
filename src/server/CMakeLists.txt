add_executable(kalkid
  main.cc
)

target_link_libraries(kalkid
  PRIVATE
    kalki::common
    kalki::api
    kalki::core
    kalki::proto
    gRPC::grpc++
    absl::flags_parse
    absl::log
)
