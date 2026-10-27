#include <new>
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#if !defined(ROCKETGPU_USE_CUDA)
#define ROCKETGPU_CAN_COMPILE_OFFICIAL_HOST_HELPERS 1
#endif

#if defined(ROCKETGPU_CAN_COMPILE_OFFICIAL_HOST_HELPERS)
#include "rocketgpu/detail/arena/mesh_path.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#endif

#include "rocketgpu/detail/arena/mesh_bvh.hpp"
#include "rocketgpu/detail/constants.hpp"
#include "rocketgpu/detail/bullet_gjk_epa2_port.hpp"
#include "rocketgpu/detail/respawn_rng.hpp"
#include "rocketgpu/detail/native_sse_normalization.hpp"

namespace rocketgpu {
namespace detail {
#include "rocketgpu/detail/step/heatseeker.hpp"
#include "rocketgpu/detail/step/snowday.hpp"
#include "rocketgpu/detail/step/arena_geometry.hpp"
#include "rocketgpu/detail/step/car_world_contacts.hpp"
#include "rocketgpu/detail/step/puck_world.hpp"
#include "rocketgpu/detail/step/arena_ball.hpp"
#include "rocketgpu/detail/step/host_validation_and_contact_debug.hpp"
#include "rocketgpu/detail/step/raycast_wheels_and_contact_prep.hpp"
#include "rocketgpu/detail/step/triangle_and_carcar.hpp"
#include "rocketgpu/detail/step/env_step.hpp"
}
}
