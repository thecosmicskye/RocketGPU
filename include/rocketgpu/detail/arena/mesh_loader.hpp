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

// Fills the per-mesh leaf grids WalkArenaMeshLeafGrid reads. A mesh whose BVH breaks the containment the grid relies
// on gets none (cell_offset -1) and keeps the walk.
void BuildArenaMeshLeafGrid(
    const std::vector<ArenaMeshBvhQuantization>& quantization,
    const std::vector<ArenaMeshBvhMeshInfo>& mesh_infos,
    const std::vector<ArenaMeshBvhNode>& nodes,
    const std::vector<ArenaMeshBvhSubtreeInfo>& subtrees,
    std::vector<ArenaMeshLeafGridInfo>& out_infos,
    std::vector<int32_t>& out_cell_start,
    std::vector<ArenaMeshLeafEntry>& out_entries);

// Sets free_half_x / free_half_y on every mesh info: the largest-area origin-centred XY box (any z) that no mesh AABB
// meets, each mesh lying beyond it in x or in y. Left 0 when some mesh straddles both axes.
void SetArenaMeshFreeBox(std::vector<ArenaMeshBvhMeshInfo>& mesh_infos);

std::vector<ArenaMeshTriangle> LoadArenaMeshTriangles(
    std::vector<ArenaMeshBvhQuantization>* out_quantization = nullptr,
    std::vector<ArenaMeshBvhMeshInfo>* out_mesh_infos = nullptr,
    std::vector<ArenaMeshBvhNode>* out_nodes = nullptr,
    std::vector<ArenaMeshBvhSubtreeInfo>* out_subtrees = nullptr,

    std::vector<ArenaMeshTriangle>* out_local_triangles = nullptr,
    int* out_local_goal_meshes = nullptr);

}
}
