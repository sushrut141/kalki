#include "kalki/common/logging.h"

#include <string>
#include <vector>

#include "absl/log/initialize.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace kalki {

void InitializeLogging() { absl::InitializeLog(); }

std::string FormatEvent(const std::string& component, const std::string& event,
                        std::initializer_list<std::pair<std::string, std::string>> fields) {
  std::vector<std::string> parts;
  parts.reserve(fields.size() + 2);
  parts.push_back(absl::StrCat("component=", component));
  parts.push_back(absl::StrCat("event=", event));
  for (const auto& [k, v] : fields) {
    parts.push_back(absl::StrCat(k, "=", v));
  }
  return absl::StrJoin(parts, " ");
}

}  // namespace kalki
