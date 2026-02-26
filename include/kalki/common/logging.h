#pragma once

#include <initializer_list>
#include <string>
#include <utility>

namespace kalki {

void InitializeLogging();
std::string FormatEvent(const std::string& component, const std::string& event,
                        std::initializer_list<std::pair<std::string, std::string>> fields = {});

}  // namespace kalki
