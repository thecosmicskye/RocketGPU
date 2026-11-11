#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "rocketgpu/detail/arena/mesh_bvh.hpp"

namespace rocketgpu {
namespace detail {
std::vector<uint16_t> BuildArenaBroadphaseCellMask(
    const std::vector<ArenaMeshTriangle>& triangles,
    const std::vector<ArenaMeshBvhMeshInfo>& mesh_infos,
    ArenaBroadphaseGrid& out_grid,
    bool fit_to_meshes = false);

void BuildArenaMeshLeafGrid(
    const std::vector<ArenaMeshBvhQuantization>& quantization,
    const std::vector<ArenaMeshBvhMeshInfo>& mesh_infos,
    const std::vector<ArenaMeshBvhNode>& nodes,
    const std::vector<ArenaMeshBvhSubtreeInfo>& subtrees,
    std::vector<ArenaMeshLeafGridInfo>& out_infos,
    std::vector<int32_t>& out_cell_start,
    std::vector<ArenaMeshLeafEntry>& out_entries,
    float cell_bt = 4.0f,
    int pad_cells_to = 1);

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

struct ArenaMeshSource {
    std::vector<Vec3> vertices;
    std::vector<int32_t> indices;
};

ArenaMeshSource ReadArenaMeshFile(const std::filesystem::path& path);

constexpr int kCustomArenaMaxMeshes = 16;
std::vector<ArenaMeshTriangle> LoadCustomArenaMeshTriangles(
    const std::vector<ArenaMeshSource>& meshes,
    std::vector<ArenaMeshBvhQuantization>& out_quantization,
    std::vector<ArenaMeshBvhMeshInfo>& out_mesh_infos,
    std::vector<ArenaMeshBvhNode>& out_nodes,
    std::vector<ArenaMeshBvhSubtreeInfo>& out_subtrees,
    std::vector<ArenaMeshTriangle>& out_local_triangles);

}
}
