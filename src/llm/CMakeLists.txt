add_library(kalki_llm SHARED
  gemini_client.cc
  local_embedding_client.cc
)
add_library(kalki::llm ALIAS kalki_llm)

target_include_directories(kalki_llm
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(kalki_llm
  PUBLIC
    kalki::common
    Llama::llama
    CURL::libcurl
    protobuf::libprotobuf
    absl::log
    absl::status
    absl::strings
    absl::synchronization
)
