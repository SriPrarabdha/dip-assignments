#pragma once
#include <string>
#include <optional>
namespace apps {
    void sharpen_experiment(const std::string& image_path, std::optional<std::string> save_dir = {});
}
