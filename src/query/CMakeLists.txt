add_library(kalki_query SHARED
  similarity.cc
  query_coordinator.cc
)
add_library(kalki::query ALIAS kalki_query)

target_include_directories(kalki_query
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(kalki_query
  PUBLIC
    kalki::common
    kalki::metadata
    kalki::storage
    kalki::proto
    Faiss::Faiss
    absl::log
    absl::status
    absl::strings
    absl::synchronization
    absl::time
)
