add_library(kalki_metadata SHARED
  metadata_store.cc
)
add_library(kalki::metadata ALIAS kalki_metadata)

target_include_directories(kalki_metadata
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(kalki_metadata
  PUBLIC
    kalki::common
    SQLite::SQLite3
    absl::log
    absl::status
    absl::strings
    absl::time
)
