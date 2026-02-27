add_library(kalki_storage SHARED
  compression.cc
  bloom_filter.cc
  wal.cc
  fresh_block.cc
  baked_block.cc
)
add_library(kalki::storage ALIAS kalki_storage)

target_include_directories(kalki_storage
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(kalki_storage
  PUBLIC
    kalki::common
    kalki::proto
    PkgConfig::ZSTD
    absl::log
    absl::status
    absl::strings
)
