#pragma once 
#include <ctime>
#include <string>

std::string get_formatted_datetime() {
    std::time_t now = std::time(nullptr);
    std::tm* tm_info = std::localtime(&now);

    char buffer[30];
    std::strftime(buffer, 30, "%Y-%m-%d %H:%M:%S", tm_info);
    return std::string(buffer);
}

std::string get_timestamp() {
  std::time_t now = std::time(nullptr);
  std::tm *tm_info = std::localtime(&now);

  char buffer[20];
  std::strftime(buffer, 20, "%Y%m%d_%H%M%S", tm_info);
  return std::string(buffer);
}