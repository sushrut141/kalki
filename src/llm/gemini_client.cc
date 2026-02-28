#include "kalki/llm/gemini_client.h"

#include <curl/curl.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/protobuf/struct.pb.h"
#include "google/protobuf/util/json_util.h"
#include "kalki/common/retry.h"

namespace kalki {

namespace {

absl::once_flag kCurlInitOnce;

size_t CurlWriteCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
  const size_t byte_count = size * nmemb;
  auto* response = static_cast<std::string*>(userdata);
  response->append(ptr, byte_count);
  return byte_count;
}

std::string StripJsonCodeFence(absl::string_view text) {
  absl::string_view trimmed = absl::StripAsciiWhitespace(text);
  if (!absl::StartsWith(trimmed, "```")) {
    return std::string(trimmed);
  }

  size_t first_newline = trimmed.find('\n');
  if (first_newline == absl::string_view::npos) {
    return std::string(trimmed);
  }

  absl::string_view inner = trimmed.substr(first_newline + 1);
  size_t last_fence = inner.rfind("```");
  if (last_fence == absl::string_view::npos) {
    return std::string(absl::StripAsciiWhitespace(inner));
  }
  return std::string(absl::StripAsciiWhitespace(inner.substr(0, last_fence)));
}

const google::protobuf::Value* FindField(const google::protobuf::Struct& message,
                                         absl::string_view field_name) {
  auto it = message.fields().find(std::string(field_name));
  if (it == message.fields().end()) {
    return nullptr;
  }
  return &it->second;
}

std::string TruncateForLog(absl::string_view value, size_t max_chars = 512) {
  if (value.size() <= max_chars) {
    return std::string(value);
  }
  return absl::StrCat(value.substr(0, max_chars), "...(truncated ", value.size(), " bytes)");
}

}  // namespace

GeminiLlmClient::GeminiLlmClient(std::string api_key, std::string model)
    : api_key_(std::move(api_key)), model_(std::move(model)) {}

std::string GeminiLlmClient::BuildPrompt(const std::string& conversation_log) const {
  return R"(
     You summarize autonomous-agent conversation logs for indexing and return an array of summaries.
      The summary should be around 2 to 3 lines.,
     1. If the conversation has 200 words or fewer, return an array with exactly one summary.,
     2. If it has more than 200 words, split it into multiple contiguous parts at meaningful
     boundaries.,
     3. Do not split in the middle of a fenced code block or at a point that loses context.,
     4. For each part generate one summary.
     5. Return a JSON array with each value being the summary for a part,
     Do this for the conversation below.
  )" + conversation_log;
}

absl::StatusOr<std::string> GeminiLlmClient::BuildRequestJson(const std::string& prompt) const {
  google::protobuf::Struct request;

  google::protobuf::Value part;
  (*part.mutable_struct_value()->mutable_fields())["text"].set_string_value(prompt);

  google::protobuf::Value parts_list;
  *parts_list.mutable_list_value()->add_values() = part;

  google::protobuf::Value content;
  (*content.mutable_struct_value()->mutable_fields())["role"].set_string_value("user");
  (*content.mutable_struct_value()->mutable_fields())["parts"] = parts_list;

  google::protobuf::Value contents_list;
  *contents_list.mutable_list_value()->add_values() = content;
  (*request.mutable_fields())["contents"] = contents_list;

  google::protobuf::Value generation_config;
  (*generation_config.mutable_struct_value()->mutable_fields())["temperature"].set_number_value(
      0.1);
  (*generation_config.mutable_struct_value()->mutable_fields())["responseMimeType"]
      .set_string_value("application/json");
  (*request.mutable_fields())["generationConfig"] = generation_config;

  std::string request_json;
  auto status = google::protobuf::util::MessageToJsonString(request, &request_json);
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("failed to serialize Gemini request JSON: ", status.ToString()));
  }
  return request_json;
}

absl::StatusOr<std::string> GeminiLlmClient::PostGenerateContent(
    const std::string& request_json) const {
  if (api_key_.empty()) {
    return absl::InvalidArgumentError("Gemini API key is empty");
  }

  absl::call_once(kCurlInitOnce,
                  []() { static_cast<void>(curl_global_init(CURL_GLOBAL_DEFAULT)); });

  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), &curl_easy_cleanup);
  if (!curl) {
    return absl::InternalError("failed to initialize CURL handle");
  }

  const std::string url = absl::StrCat("https://generativelanguage.googleapis.com/v1beta/models/",
                                       model_, ":generateContent?key=", api_key_);

  std::string response_body;
  const absl::Time start = absl::Now();

  curl_slist* header_list = nullptr;
  header_list = curl_slist_append(header_list, "Content-Type: application/json");
  if (header_list == nullptr) {
    return absl::InternalError("failed to allocate CURL header list");
  }
  std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(header_list,
                                                                      &curl_slist_free_all);

  curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl.get(), CURLOPT_POST, 1L);
  curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers.get());
  curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS, request_json.c_str());
  curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDSIZE, static_cast<long>(request_json.size()));
  curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, CurlWriteCallback);
  curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &response_body);
  curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS, 30000L);

  const CURLcode perform_code = curl_easy_perform(curl.get());
  if (perform_code != CURLE_OK) {
    return absl::InternalError(
        absl::StrCat("Gemini request failed: ", curl_easy_strerror(perform_code)));
  }

  long http_code = 0;
  curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);
  const int64_t latency_ms = absl::ToInt64Milliseconds(absl::Now() - start);
  DLOG(INFO) << "component=gemini event=http_response model=" << model_
             << " code=" << http_code << " latency_ms=" << latency_ms
             << " response_bytes=" << response_body.size()
             << " body_snippet=" << TruncateForLog(response_body);
  if (http_code < 200 || http_code >= 300) {
    return absl::InternalError(
        absl::StrCat("Gemini HTTP error ", http_code, " body=", response_body));
  }

  return response_body;
}

absl::StatusOr<std::string> GeminiLlmClient::ExtractCandidateText(
    const std::string& response_json) const {
  google::protobuf::Struct response;
  auto parse_status = google::protobuf::util::JsonStringToMessage(response_json, &response);
  if (!parse_status.ok()) {
    return absl::InternalError(
        absl::StrCat("failed to parse Gemini response JSON: ", parse_status.ToString()));
  }

  const google::protobuf::Value* candidates = FindField(response, "candidates");
  if (candidates == nullptr || !candidates->has_list_value() ||
      candidates->list_value().values().empty()) {
    return absl::InternalError("Gemini response has no candidates");
  }

  std::vector<std::string> parts;
  for (const google::protobuf::Value& candidate_value : candidates->list_value().values()) {
    if (!candidate_value.has_struct_value()) {
      continue;
    }
    const google::protobuf::Value* content = FindField(candidate_value.struct_value(), "content");
    if (content == nullptr || !content->has_struct_value()) {
      continue;
    }
    const google::protobuf::Value* content_parts = FindField(content->struct_value(), "parts");
    if (content_parts == nullptr || !content_parts->has_list_value()) {
      continue;
    }

    for (const google::protobuf::Value& part_value : content_parts->list_value().values()) {
      if (!part_value.has_struct_value()) {
        continue;
      }
      const google::protobuf::Value* text_value = FindField(part_value.struct_value(), "text");
      if (text_value != nullptr && text_value->has_string_value()) {
        parts.push_back(text_value->string_value());
      }
    }
  }

  if (parts.empty()) {
    return absl::InternalError("Gemini response candidates have no text parts");
  }
  DLOG(INFO) << "component=gemini event=candidate_text_extracted parts=" << parts.size()
             << " text_snippet=" << TruncateForLog(absl::StrJoin(parts, "\n"));
  return absl::StrJoin(parts, "\n");
}

absl::StatusOr<std::vector<std::string>> GeminiLlmClient::ParseSummaryPayload(
    const std::string& model_text) const {
  const std::string payload_json = StripJsonCodeFence(model_text);
  google::protobuf::Value payload;
  auto parse_status = google::protobuf::util::JsonStringToMessage(payload_json, &payload);
  if (!parse_status.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("invalid response: malformed JSON payload: ", parse_status.ToString()));
  }

  if (!payload.has_list_value()) {
    return absl::InvalidArgumentError("invalid response: expected top-level JSON array of strings");
  }

  const google::protobuf::ListValue& summary_list = payload.list_value();
  if (summary_list.values().empty()) {
    return absl::InvalidArgumentError("invalid response: summary array is empty");
  }

  std::vector<std::string> summaries;
  summaries.reserve(summary_list.values().size());
  for (const google::protobuf::Value& summary_value : summary_list.values()) {
    if (!summary_value.has_string_value()) {
      return absl::InvalidArgumentError(
          "invalid response: summary array contains non-string entry");
    }
    const std::string summary =
        std::string(absl::StripAsciiWhitespace(summary_value.string_value()));
    if (summary.empty()) {
      return absl::InvalidArgumentError("invalid response: summary array contains empty string");
    }
    summaries.push_back(summary);
  }
  return summaries;
}

absl::StatusOr<std::vector<std::string>> GeminiLlmClient::SummarizeConversation(
    const std::string& conversation_log) {
  if (conversation_log.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("Conversation log is empty"));
  }

  const std::string prompt = BuildPrompt(conversation_log);
  DLOG(INFO) << "component=gemini event=prompt_built prompt_bytes=" << prompt.size();

  auto request_json_or = BuildRequestJson(prompt);
  if (!request_json_or.ok()) {
    return request_json_or.status();
  }

  const RetryOptions retry_options{
      .max_attempts = 3,
      .initial_backoff = absl::Milliseconds(100),
      .backoff_multiplier = 2.0,
      .max_backoff = absl::Milliseconds(500),
  };
  int attempt = 0;
  auto summaries_or = RetryStatusOr(
      [&]() -> absl::StatusOr<std::vector<std::string>> {
        ++attempt;
        DLOG(INFO) << "component=gemini event=attempt_start model=" << model_
                   << " attempt=" << attempt;
        auto response_or = PostGenerateContent(*request_json_or);
        if (!response_or.ok()) {
          DLOG(INFO) << "component=gemini event=attempt_failed model=" << model_
                     << " attempt=" << attempt << " status=" << response_or.status();
          return response_or.status();
        }

        auto candidate_text_or = ExtractCandidateText(*response_or);
        if (!candidate_text_or.ok()) {
          DLOG(INFO) << "component=gemini event=attempt_failed model=" << model_
                     << " attempt=" << attempt << " status=" << candidate_text_or.status();
          return candidate_text_or.status();
        }
        auto parsed_or = ParseSummaryPayload(*candidate_text_or);
        if (!parsed_or.ok()) {
          DLOG(INFO) << "component=gemini event=attempt_failed model=" << model_
                     << " attempt=" << attempt << " status=" << parsed_or.status();
          return parsed_or.status();
        }
        DLOG(INFO) << "component=gemini event=attempt_success model=" << model_
                   << " attempt=" << attempt << " summaries=" << parsed_or->size();
        return parsed_or;
      },
      [](const absl::Status& status) {
        return status.code() == absl::StatusCode::kInvalidArgument ||
               status.code() == absl::StatusCode::kInternal ||
               status.code() == absl::StatusCode::kUnavailable ||
               status.code() == absl::StatusCode::kDeadlineExceeded;
      },
      retry_options);
  if (!summaries_or.ok()) {
    LOG(ERROR) << "component=gemini event=summarize_retry_exhausted status="
               << summaries_or.status();
    return summaries_or.status();
  }

  LOG(INFO) << "component=gemini event=summarize model=" << model_
            << " summary_parts=" << summaries_or->size();
  return summaries_or;
}

}  // namespace kalki
