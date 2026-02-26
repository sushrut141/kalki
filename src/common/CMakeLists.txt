add_library(kalki_common SHARED
  config.cc
  logging.cc
  thread_pool.cc
)
add_library(kalki::common ALIAS kalki_common)

target_include_directories(kalki_common
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(kalki_common
  PUBLIC
    absl::flags
    absl::flags_parse
    absl::log
    absl::log_initialize
    absl::strings
    absl::status
    absl::statusor
    absl::synchronization
    absl::time
)
