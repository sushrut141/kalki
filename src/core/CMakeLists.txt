add_library(kalki_core SHARED
  database_engine.cc
)
add_library(kalki::core ALIAS kalki_core)

target_include_directories(kalki_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(kalki_core
  PUBLIC
    kalki::common
    kalki::metadata
    kalki::storage
    kalki::llm
    kalki::workers
    kalki::query
    kalki::proto
    absl::log
    absl::status
    absl::strings
    absl::time
)
