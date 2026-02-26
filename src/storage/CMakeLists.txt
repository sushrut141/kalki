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
    absl::log
    absl::status
    absl::strings
)

if(ZSTD_FOUND)
  target_include_directories(kalki_storage PUBLIC ${ZSTD_INCLUDE_DIRS})
  target_link_directories(kalki_storage PUBLIC ${ZSTD_LIBRARY_DIRS})
  target_link_libraries(kalki_storage PUBLIC ${ZSTD_LIBRARIES})
  target_compile_definitions(kalki_storage PUBLIC KALKI_HAVE_ZSTD)
endif()
