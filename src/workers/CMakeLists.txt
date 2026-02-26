add_library(kalki_workers SHARED
  ingestion_worker.cc
  compaction_worker.cc
)
add_library(kalki::workers ALIAS kalki_workers)

target_include_directories(kalki_workers
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(kalki_workers
  PUBLIC
    kalki::common
    kalki::metadata
    kalki::storage
    kalki::llm
    kalki::proto
    absl::log
    absl::status
    absl::strings
    absl::time
)
