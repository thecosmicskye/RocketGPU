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

void BuildArenaMeshLeafGrid(
    const std::vector<ArenaMeshBvhQuantization>& quantization,
    const std::vector<ArenaMeshBvhMeshInfo>& mesh_infos,
    const std::vector<ArenaMeshBvhNode>& nodes,
    const std::vector<ArenaMeshBvhSubtreeInfo>& subtrees,
    std::vector<ArenaMeshLeafGridInfo>& out_infos,
    std::vector<int32_t>& out_cell_start,
    std::vector<ArenaMeshLeafEntry>& out_entries,
    float cell_bt = 4.0f);

// Fills out (size nx * ny * nz, x fastest) for ArenaMeshClearance over every triangle of mesh and, where
// cal_origin_sign != 0, of local_mesh shifted by its goal origin. Runs on up to threads threads.
void BuildArenaMeshClearance(
    const std::vector<ArenaMeshTriangle>& mesh,
    const std::vector<ArenaMeshTriangle>& local_mesh,
    std::vector<uint8_t>& out,
    Vec3& out_lo,
    float& out_inv_cell,
    float& out_unit,
    int32_t out_dims[3],
    int threads = 1);

void SetArenaMeshFreeBox(std::vector<ArenaMeshBvhMeshInfo>& mesh_infos);

void SetArenaMeshRayOrder(std::vector<ArenaMeshBvhMeshInfo>& mesh_infos, float neg_x_plane_max_x);

std::vector<ArenaMeshTriangle> LoadArenaMeshTriangles(
    std::vector<ArenaMeshBvhQuantization>* out_quantization = nullptr,
    std::vector<ArenaMeshBvhMeshInfo>* out_mesh_infos = nullptr,
    std::vector<ArenaMeshBvhNode>* out_nodes = nullptr,
    std::vector<ArenaMeshBvhSubtreeInfo>* out_subtrees = nullptr,

    std::vector<ArenaMeshTriangle>* out_local_triangles = nullptr,
    int* out_local_goal_meshes = nullptr);

}
}
