#pragma once

#include <cstdint>
#include <vector>

#include "rocketgpu/detail/arena/mesh_bvh.hpp"

namespace rocketgpu {
namespace detail {
std::vector<uint16_t> BuildArenaBroadphaseCellMask(
    const std::vector<ArenaMeshTriangle>& triangles,
    const std::vector<ArenaMeshBvhMeshInfo>& mesh_infos,
    ArenaBroadphaseGrid& out_grid);

std::vector<ArenaMeshTriangle> LoadArenaMeshTriangles(
    std::vector<ArenaMeshBvhQuantization>* out_quantization = nullptr,
    std::vector<ArenaMeshBvhMeshInfo>* out_mesh_infos = nullptr,
    std::vector<ArenaMeshBvhNode>* out_nodes = nullptr,
    std::vector<ArenaMeshBvhSubtreeInfo>* out_subtrees = nullptr,

    std::vector<ArenaMeshTriangle>* out_local_triangles = nullptr,
    int* out_local_goal_meshes = nullptr);

bool HasGoalMeshes();
}
}
