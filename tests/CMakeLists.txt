find_package(GTest CONFIG REQUIRED)

add_executable(kalki_gemini_client_test
  gemini_client_test.cc
)

target_link_libraries(kalki_gemini_client_test
  PRIVATE
    kalki::llm
    GTest::gtest_main
)

add_test(NAME kalki_gemini_client_test COMMAND kalki_gemini_client_test)
