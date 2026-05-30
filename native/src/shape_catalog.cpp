#include "minescript_miner/shape_catalog.hpp"

#include <cstddef>
#include <vector>

namespace minescript_miner {

static void register_shape(std::vector<std::string> &names, const std::string &name) {
    names.push_back(name);
}

static std::string connection_mask_name(int mask) {
    static constexpr const char *directions[] = {"north", "east", "south", "west"};

    std::string name;
    for (int bit = 0; bit < 4; ++bit) {
        if ((mask & (1 << bit)) == 0) {
            continue;
        }
        if (!name.empty()) {
            name += "_";
        }
        name += directions[bit];
    }
    return name.empty() ? "none" : name;
}

const std::vector<std::string> &shape_names() {
    static const std::vector<std::string> names = [] {
        static constexpr const char *directions[] = {"north", "east", "south", "west"};
        static constexpr const char *halves[] = {"bottom", "top"};
        static constexpr const char *stair_shapes[] = {
            "straight",
            "inner_left",
            "inner_right",
            "outer_left",
            "outer_right",
        };

        std::vector<std::string> generated;
        register_shape(generated, "empty");
        register_shape(generated, "full_cube");
        register_shape(generated, "oak_slab_bottom");
        register_shape(generated, "oak_slab_top");

        for (const char *direction : directions) {
            for (const char *half : halves) {
                for (const char *stair_shape : stair_shapes) {
                    register_shape(
                        generated,
                        std::string("oak_stairs_") +
                            direction + "_" +
                            half + "_" +
                            stair_shape
                    );
                }
            }
        }

        for (int mask = 0; mask < 16; ++mask) {
            register_shape(generated, "pane_" + connection_mask_name(mask));
        }

        for (int mask = 0; mask < 16; ++mask) {
            register_shape(generated, "oak_fence_" + connection_mask_name(mask));
        }

        return generated;
    }();

    return names;
}

std::string shape_id_name(std::int32_t shape_id) {
    const auto &names = shape_names();
    if (shape_id < 0 || static_cast<std::size_t>(shape_id) >= names.size()) {
        return "unknown_shape";
    }
    return names[static_cast<std::size_t>(shape_id)];
}

std::int32_t shape_count() {
    return static_cast<std::int32_t>(shape_names().size());
}

}
