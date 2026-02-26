#include "kalki/llm/gemini_client.h"

#include <string>
#include <vector>

#include "../../../../../opt/homebrew/Cellar/abseil/20260107.1/include/absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

namespace {

std::string BuildSmallConversation() {
  std::string text;
  text.reserve(16 * 1024);

  text += "Agent: starting debug session for query planner.\\n";
  text += "```cpp\\n";
  text += "for (int i = 0; i < workers; ++i) { schedule(i); }\\n";
  text += "```\\n";

  for (int i = 0; i < 620; ++i) {
    text += absl::StrCat("token", i, " ");
  }

  text += "\\nAgent: finished collecting diagnostics and next actions.\\n";
  return text;
}

std::string BuildLongConversation() {
  return R"(
    GoogleTest helps you write better C++ tests.
    GoogleTest is a testing framework developed by the Testing Technology team with Google’s specific requirements and constraints in mind. Whether you work on Linux, Windows, or a Mac, if you write C++ code, GoogleTest can help you. And it supports any kind of tests, not just unit tests.

    So what makes a good test, and how does GoogleTest fit in? We believe:

    Tests should be independent and repeatable. It’s a pain to debug a test that succeeds or fails as a result of other tests. GoogleTest isolates the tests by running each of them on a different object. When a test fails, GoogleTest allows you to run it in isolation for quick debugging.
    Tests should be well organized and reflect the structure of the tested code. GoogleTest groups related tests into test suites that can share data and subroutines. This common pattern is easy to recognize and makes tests easy to maintain. Such consistency is especially helpful when people switch projects and start to work on a new code base.
    Tests should be portable and reusable. Google has a lot of code that is platform-neutral; its tests should also be platform-neutral. GoogleTest works on different OSes, with different compilers, with or without exceptions, so GoogleTest tests can work with a variety of configurations.
    When tests fail, they should provide as much information about the problem as possible. GoogleTest doesn’t stop at the first test failure. Instead, it only stops the current test and continues with the next. You can also set up tests that report non-fatal failures after which the current test continues. Thus, you can detect and fix multiple bugs in a single run-edit-compile cycle.
    The testing framework should liberate test writers from housekeeping chores and let them focus on the test content. GoogleTest automatically keeps track of all tests defined, and doesn’t require the user to enumerate them in order to run them.
    Tests should be fast. With GoogleTest, you can reuse shared resources across tests and pay for the set-up/tear-down only once, without making tests depend on each other.
    Since GoogleTest is based on the popular xUnit architecture, you’ll feel right at home if you’ve used JUnit or PyUnit before. If not, it will take you about 10 minutes to learn the basics and get started. So let’s go!
  )";
}

}  // namespace

TEST(GeminiClientTest, SummarizeSmallConversationReturnsSinglePart) {
  static constexpr char kHardcodedApiKey[] = "AIzaSyCz_NKmRYcet6A9e22WSNpJOR6SjUxOXSE";
  static constexpr char kModel[] = "gemini-3-flash-preview";
  kalki::GeminiLlmClient client(kHardcodedApiKey, kModel);
  const std::string conversation = BuildSmallConversation();

  auto result_or =
      client.SummarizeConversation(conversation);
  ASSERT_TRUE(result_or.ok()) << result_or.status();
  const std::vector<std::string>& summaries = *result_or;

  ASSERT_EQ(1, summaries.size());
}

TEST(GeminiClientTest, SummarizeLongConversationHasMultipleParts) {
  static constexpr char kHardcodedApiKey[] = "AIzaSyCz_NKmRYcet6A9e22WSNpJOR6SjUxOXSE";
  static constexpr char kModel[] = "gemini-3-flash-preview";
  kalki::GeminiLlmClient client(kHardcodedApiKey, kModel);
  const std::string conversation = BuildLongConversation();

  auto result_or =
      client.SummarizeConversation(conversation);
  ASSERT_TRUE(result_or.ok()) << result_or.status();
  const std::vector<std::string>& summaries = *result_or;

  LOG(INFO) << "Obtained summaries as [0] " << summaries[0] << " \n [1] " << summaries[1];
  ASSERT_LT(1, summaries.size());
}
