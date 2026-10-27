#pragma once

#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace rocketgpu {
namespace detail {
inline std::filesystem::path OfficialCollisionMeshRoot() {
    const char* override_path = std::getenv("ROCKETGPU_COLLISION_MESH_DIR");
    if (override_path && *override_path) {
        return std::filesystem::path(override_path);
    }
    throw std::runtime_error("set ROCKETGPU_COLLISION_MESH_DIR to the RocketSim collision_meshes directory");
}
}
}
