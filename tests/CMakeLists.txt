find_package(GTest CONFIG REQUIRED)

    add_executable(kalki_gemini_client_test gemini_client_test.cc)

    target_link_libraries(kalki_gemini_client_test PRIVATE kalki::llm GTest::gtest_main)

    add_test(NAME kalki_gemini_client_test COMMAND kalki_gemini_client_test)

    add_executable(kalki_database_e2e_test database_e2e_test.cc)

    target_link_libraries(kalki_database_e2e_test PRIVATE
                                          kalki::core kalki::llm absl::time GTest::gtest_main)

    add_test(NAME kalki_database_e2e_test COMMAND kalki_database_e2e_test)
