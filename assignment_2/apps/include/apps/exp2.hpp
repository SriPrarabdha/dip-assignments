#pragma once
#include <string>
#include <optional>

namespace apps {
    void scale_rotate_experiment(const std::string& image_path, std::optional<std::string> save_dir = {});
}
