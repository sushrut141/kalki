find_package(GTest CONFIG REQUIRED)

# Disabled temporarily: relies on external Gemini network/API availability.
# add_executable(kalki_gemini_client_test gemini_client_test.cc)
# target_link_libraries(kalki_gemini_client_test PRIVATE kalki::llm GTest::gtest_main)
# add_test(NAME kalki_gemini_client_test COMMAND kalki_gemini_client_test)

add_executable(kalki_local_embedding_client_test local_embedding_client_test.cc)
target_compile_definitions(kalki_local_embedding_client_test
  PRIVATE KALKI_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(kalki_local_embedding_client_test PRIVATE kalki::llm GTest::gtest_main)
add_test(NAME kalki_local_embedding_client_test COMMAND kalki_local_embedding_client_test)

add_executable(kalki_database_e2e_test database_e2e_test.cc)
target_link_libraries(kalki_database_e2e_test PRIVATE
  kalki::core kalki::llm absl::time GTest::gtest_main)
add_test(NAME kalki_database_e2e_test COMMAND kalki_database_e2e_test)

add_executable(kalki_block_lifecycle_test block_lifecycle_test.cc)
target_link_libraries(kalki_block_lifecycle_test PRIVATE
  kalki::core kalki::metadata absl::time GTest::gtest_main)
add_test(NAME kalki_block_lifecycle_test COMMAND kalki_block_lifecycle_test)

add_executable(kalki_query_behavior_test query_behavior_test.cc)
target_link_libraries(kalki_query_behavior_test PRIVATE
  kalki::core kalki::metadata absl::time GTest::gtest_main)
add_test(NAME kalki_query_behavior_test COMMAND kalki_query_behavior_test)

add_executable(kalki_metadata_store_test metadata_store_test.cc)
target_link_libraries(kalki_metadata_store_test PRIVATE
  kalki::core kalki::metadata absl::time GTest::gtest_main)
add_test(NAME kalki_metadata_store_test COMMAND kalki_metadata_store_test)
