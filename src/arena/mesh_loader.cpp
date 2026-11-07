#include "rocketgpu/detail/arena/mesh_loader.hpp"
#include "rocketgpu/detail/arena/mesh_path.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

namespace rocketgpu {
namespace detail {
namespace {
struct TriangleIndex {
    int32_t vertex_index[3];
};

struct MeshVertex {
    float x;
    float y;
    float z;
};

float GetInternalEdgeAngle(const Vec3& ref_axis0, const Vec3& ref_axis1, const Vec3& swing_axis) {
    return RocketGpuUcrtAtan2f(Dot(swing_axis, ref_axis0), Dot(swing_axis, ref_axis1));
}

static Vec3 SseNormalizeVec3(const Vec3& v, const Vec3& fallback) {
    const float len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len_sq <= 1e-12f) {
        return fallback;
    }
    const float inv_len = 1.0f / sqrtf(len_sq);
    return Vec3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}

Vec3 SubVec3(const Vec3& a, const Vec3& b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vec3 AddVec3(const Vec3& a, const Vec3& b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vec3 ScaleVec3(const Vec3& v, float s) {
    return Vec3(v.x * s, v.y * s, v.z * s);
}

Vec3 NegVec3(const Vec3& v) {
    return Vec3(-v.x, -v.y, -v.z);
}

void PopulateTriangleInternalEdgeInfo(
    const std::vector<TriangleIndex>& tri_indices,
    std::vector<ArenaMeshTriangle>& triangles,
    int tri_a_idx,
    int tri_b_idx,
    int mesh_base) {
    (void)tri_indices;
    (void)mesh_base;

    constexpr float kEqualVertexThresholdSq = 1.0e-4f * 1.0e-4f;
    const ArenaMeshTriangle& tri_a_ref = triangles[static_cast<size_t>(tri_a_idx)];
    const ArenaMeshTriangle& tri_b_ref = triangles[static_cast<size_t>(tri_b_idx)];
    const Vec3 verts_a[3] = {tri_a_ref.a, tri_a_ref.b, tri_a_ref.c};
    const Vec3 verts_b[3] = {tri_b_ref.a, tri_b_ref.b, tri_b_ref.c};

    int shared_verts_a[2] = {-1, -1};
    int shared_verts_b[2] = {-1, -1};
    int shared_count = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const Vec3 delta = SubVec3(verts_a[i], verts_b[j]);
            if (LengthSq(delta) < kEqualVertexThresholdSq) {
                shared_verts_a[shared_count] = i;
                shared_verts_b[shared_count] = j;
                ++shared_count;
                if (shared_count >= 3) {
                    return;
                }
                break;
            }
        }
        if (shared_count >= 3) {
            return;
        }
    }
    if (shared_count != 2) {
        return;
    }

    if (shared_verts_a[0] == 0 && shared_verts_a[1] == 2) {
        std::swap(shared_verts_a[0], shared_verts_a[1]);
        std::swap(shared_verts_b[0], shared_verts_b[1]);
    }

    ArenaMeshTriangle& tri_a = triangles[tri_a_idx];
    const ArenaMeshTriangle& tri_b = triangles[tri_b_idx];
    const Vec3 tri_a_vertices[3] = {tri_a.a, tri_a.b, tri_a.c};
    const Vec3 tri_b_vertices[3] = {tri_b.a, tri_b.b, tri_b.c};

    const int sum_verts_a = shared_verts_a[0] + shared_verts_a[1];
    const int other_index_a = 3 - sum_verts_a;
    const int other_index_b = 3 - (shared_verts_b[0] + shared_verts_b[1]);

    Vec3 edge = SubVec3(tri_a_vertices[shared_verts_a[1]], tri_a_vertices[shared_verts_a[0]]);
    if (LengthSq(edge) <= 1.0e-12f) {
        return;
    }
    edge = SseNormalizeVec3(edge, Vec3(1.0f, 0.0f, 0.0f));

    const Vec3 normal_a = SseNormalizeVec3(
        Cross(SubVec3(tri_a_vertices[1], tri_a_vertices[0]),
              SubVec3(tri_a_vertices[2], tri_a_vertices[0])),
        Vec3(0.0f, 0.0f, 1.0f));
    const Vec3 tri_b0 = tri_b_vertices[shared_verts_b[1]];
    const Vec3 tri_b1 = tri_b_vertices[shared_verts_b[0]];
    const Vec3 tri_b2 = tri_b_vertices[other_index_b];
    Vec3 normal_b = SseNormalizeVec3(
        Cross(SubVec3(tri_b1, tri_b0), SubVec3(tri_b2, tri_b0)),
        Normalize(tri_b.normal, Vec3(0.0f, 0.0f, 1.0f)));

    Vec3 edge_cross_a = SseNormalizeVec3(Cross(edge, normal_a), normal_a);
    if (Dot(edge_cross_a, SubVec3(tri_a_vertices[other_index_a], tri_a_vertices[shared_verts_a[0]])) < 0.0f) {
        edge_cross_a = NegVec3(edge_cross_a);
    }

    Vec3 edge_cross_b = SseNormalizeVec3(Cross(edge, normal_b), normal_b);
    if (Dot(edge_cross_b, SubVec3(tri_b_vertices[other_index_b], tri_b_vertices[shared_verts_b[0]])) < 0.0f) {
        edge_cross_b = NegVec3(edge_cross_b);
    }

    float corrected_angle = 0.0f;
    bool is_convex = false;
    const Vec3 calculated_edge = Cross(edge_cross_a, edge_cross_b);
    if (LengthSq(calculated_edge) >= 0.0001f) {
        const Vec3 calculated_edge_unit = SseNormalizeVec3(calculated_edge, edge);
        const Vec3 calculated_normal_a = SseNormalizeVec3(Cross(calculated_edge_unit, edge_cross_a), normal_a);
        const float angle2 = GetInternalEdgeAngle(calculated_normal_a, edge_cross_a, edge_cross_b);
        const float ang4 = Pi() - angle2;
        is_convex = Dot(normal_a, edge_cross_b) < 0.0f;
        corrected_angle = is_convex ? ang4 : -ang4;
    }

    auto set_edge_info = [&](float& edge_angle, uint8_t convex_flag, uint8_t swap_flag, const Vec3& edge_axis) {
        Vec3 computed_normal_b = BulletQuatRotate(
            BulletQuatFromAxisAngle(edge_axis, -corrected_angle),
            normal_a);
        if (Dot(computed_normal_b, normal_b) < 0.0f) {
            computed_normal_b = NegVec3(computed_normal_b);
            tri_a.edge_flags = static_cast<uint8_t>(tri_a.edge_flags | swap_flag);
        }
        edge_angle = -corrected_angle;
        if (is_convex) {
            tri_a.edge_flags = static_cast<uint8_t>(tri_a.edge_flags | convex_flag);
        }
    };

    switch (sum_verts_a) {
        case 1:
            set_edge_info(
                tri_a.edge_v0v1_angle,
                ROCKETGPU_TRI_INFO_V0V1_CONVEX,
                ROCKETGPU_TRI_INFO_V0V1_SWAP_NORMALB,
                SubVec3(tri_a_vertices[0], tri_a_vertices[1]));
            break;
        case 2:
            set_edge_info(
                tri_a.edge_v2v0_angle,
                ROCKETGPU_TRI_INFO_V2V0_CONVEX,
                ROCKETGPU_TRI_INFO_V2V0_SWAP_NORMALB,
                SubVec3(tri_a_vertices[2], tri_a_vertices[0]));
            break;
        case 3:
            set_edge_info(
                tri_a.edge_v1v2_angle,
                ROCKETGPU_TRI_INFO_V1V2_CONVEX,
                ROCKETGPU_TRI_INFO_V1V2_SWAP_NORMALB,
                SubVec3(tri_a_vertices[1], tri_a_vertices[2]));
            break;
        default:
            break;
    }
}

constexpr int kArenaMeshMaxSubtreeSizeBytes = 2048;
constexpr float kArenaMeshMinAabbDimension = 0.002f;
constexpr float kArenaMeshMinAabbHalfDimension = 0.001f;

int ArenaMeshBvhMaxAxis(const Vec3& v) {
    return v.x < v.y ? (v.y < v.z ? 2 : 1) : (v.x < v.z ? 2 : 0);
}

Vec3 GetArenaMeshBvhNodeAabbMin(
    const ArenaMeshBvhQuantization& quantization,
    const ArenaMeshBvhNode& node) {
    return UnquantizeArenaMeshBvhPoint(quantization, node.quantized_aabb_min);
}

Vec3 GetArenaMeshBvhNodeAabbMax(
    const ArenaMeshBvhQuantization& quantization,
    const ArenaMeshBvhNode& node) {
    return UnquantizeArenaMeshBvhPoint(quantization, node.quantized_aabb_max);
}

void BuildArenaMeshBulletStyleBvh(
    const std::vector<ArenaMeshTriangle>& triangles,
    int32_t mesh_triangle_start,
    int32_t mesh_triangle_count,
    const ArenaMeshBvhQuantization& quantization,
    std::vector<ArenaMeshBvhNode>& out_nodes,
    std::vector<ArenaMeshBvhSubtreeInfo>& out_subtrees) {
    std::vector<ArenaMeshBvhNode> leaf_nodes(static_cast<size_t>(mesh_triangle_count));
    for (int32_t local_tri_idx = 0; local_tri_idx < mesh_triangle_count; ++local_tri_idx) {
        const ArenaMeshTriangle& tri = triangles[static_cast<size_t>(mesh_triangle_start + local_tri_idx)];
        Vec3 aabb_min = tri.min;
        Vec3 aabb_max = tri.max;
        if (aabb_max.x - aabb_min.x < kArenaMeshMinAabbDimension) {
            aabb_max.x += kArenaMeshMinAabbHalfDimension;
            aabb_min.x -= kArenaMeshMinAabbHalfDimension;
        }
        if (aabb_max.y - aabb_min.y < kArenaMeshMinAabbDimension) {
            aabb_max.y += kArenaMeshMinAabbHalfDimension;
            aabb_min.y -= kArenaMeshMinAabbHalfDimension;
        }
        if (aabb_max.z - aabb_min.z < kArenaMeshMinAabbDimension) {
            aabb_max.z += kArenaMeshMinAabbHalfDimension;
            aabb_min.z -= kArenaMeshMinAabbHalfDimension;
        }

        ArenaMeshBvhNode& leaf = leaf_nodes[static_cast<size_t>(local_tri_idx)];
        QuantizeArenaMeshBvhPoint(quantization, aabb_min, 0, leaf.quantized_aabb_min);
        QuantizeArenaMeshBvhPoint(quantization, aabb_max, 1, leaf.quantized_aabb_max);
        leaf.escape_index_or_triangle_index = local_tri_idx;
    }

    out_nodes.assign(static_cast<size_t>(2 * mesh_triangle_count), ArenaMeshBvhNode{});
    out_subtrees.clear();
    int cur_node_index = 0;

    auto calc_splitting_axis = [&](int start_index, int end_index) {
        Vec3 means(0.0f, 0.0f, 0.0f);
        Vec3 variance(0.0f, 0.0f, 0.0f);
        const int num_indices = end_index - start_index;
        for (int i = start_index; i < end_index; ++i) {
            const Vec3 center = ScaleVec3(
                AddVec3(
                    GetArenaMeshBvhNodeAabbMax(quantization, leaf_nodes[static_cast<size_t>(i)]),
                    GetArenaMeshBvhNodeAabbMin(quantization, leaf_nodes[static_cast<size_t>(i)])),
                0.5f);
            means = AddVec3(means, center);
        }
        means = ScaleVec3(means, 1.0f / static_cast<float>(num_indices));

        for (int i = start_index; i < end_index; ++i) {
            const Vec3 center = ScaleVec3(
                AddVec3(
                    GetArenaMeshBvhNodeAabbMax(quantization, leaf_nodes[static_cast<size_t>(i)]),
                    GetArenaMeshBvhNodeAabbMin(quantization, leaf_nodes[static_cast<size_t>(i)])),
                0.5f);
            Vec3 diff2 = SubVec3(center, means);
            diff2 = MulComponents(diff2, diff2);
            variance = AddVec3(variance, diff2);
        }
        variance = ScaleVec3(variance, 1.0f / static_cast<float>(num_indices - 1));
        return ArenaMeshBvhMaxAxis(variance);
    };

    auto sort_and_calc_splitting_index = [&](int start_index, int end_index, int split_axis) {
        int split_index = start_index;
        const int num_indices = end_index - start_index;
        Vec3 means(0.0f, 0.0f, 0.0f);
        for (int i = start_index; i < end_index; ++i) {
            const Vec3 center = ScaleVec3(
                AddVec3(
                    GetArenaMeshBvhNodeAabbMax(quantization, leaf_nodes[static_cast<size_t>(i)]),
                    GetArenaMeshBvhNodeAabbMin(quantization, leaf_nodes[static_cast<size_t>(i)])),
                0.5f);
            means = AddVec3(means, center);
        }
        means = ScaleVec3(means, 1.0f / static_cast<float>(num_indices));
        const float split_value = (&means.x)[split_axis];

        for (int i = start_index; i < end_index; ++i) {
            const Vec3 center = ScaleVec3(
                AddVec3(
                    GetArenaMeshBvhNodeAabbMax(quantization, leaf_nodes[static_cast<size_t>(i)]),
                    GetArenaMeshBvhNodeAabbMin(quantization, leaf_nodes[static_cast<size_t>(i)])),
                0.5f);
            if ((&center.x)[split_axis] > split_value) {
                std::swap(leaf_nodes[static_cast<size_t>(i)], leaf_nodes[static_cast<size_t>(split_index)]);
                ++split_index;
            }
        }

        const int range_balanced_indices = num_indices / 3;
        const bool unbalanced =
            split_index <= (start_index + range_balanced_indices) ||
            split_index >= (end_index - 1 - range_balanced_indices);
        if (unbalanced) {
            split_index = start_index + (num_indices >> 1);
        }
        return split_index;
    };

    auto update_subtree_headers = [&](int left_child_node_index, int right_child_node_index) {
        const ArenaMeshBvhNode& left_child = out_nodes[static_cast<size_t>(left_child_node_index)];
        const int left_subtree_size = left_child.is_leaf_node() ? 1 : left_child.get_escape_index();
        const int left_subtree_size_in_bytes = left_subtree_size * static_cast<int>(sizeof(ArenaMeshBvhNode));

        const ArenaMeshBvhNode& right_child = out_nodes[static_cast<size_t>(right_child_node_index)];
        const int right_subtree_size = right_child.is_leaf_node() ? 1 : right_child.get_escape_index();
        const int right_subtree_size_in_bytes = right_subtree_size * static_cast<int>(sizeof(ArenaMeshBvhNode));

        if (left_subtree_size_in_bytes <= kArenaMeshMaxSubtreeSizeBytes) {
            ArenaMeshBvhSubtreeInfo subtree;
            std::copy_n(left_child.quantized_aabb_min, 3, subtree.quantized_aabb_min);
            std::copy_n(left_child.quantized_aabb_max, 3, subtree.quantized_aabb_max);
            subtree.root_node_index = left_child_node_index;
            subtree.subtree_size = left_subtree_size;
            out_subtrees.push_back(subtree);
        }

        if (right_subtree_size_in_bytes <= kArenaMeshMaxSubtreeSizeBytes) {
            ArenaMeshBvhSubtreeInfo subtree;
            std::copy_n(right_child.quantized_aabb_min, 3, subtree.quantized_aabb_min);
            std::copy_n(right_child.quantized_aabb_max, 3, subtree.quantized_aabb_max);
            subtree.root_node_index = right_child_node_index;
            subtree.subtree_size = right_subtree_size;
            out_subtrees.push_back(subtree);
        }
    };

    auto build_tree = [&](auto&& self, int start_index, int end_index) -> void {
        const int num_indices = end_index - start_index;
        const int cur_index = cur_node_index;
        if (num_indices == 1) {
            out_nodes[static_cast<size_t>(cur_node_index)] = leaf_nodes[static_cast<size_t>(start_index)];
            ++cur_node_index;
            return;
        }

        const int split_axis = calc_splitting_axis(start_index, end_index);
        const int split_index = sort_and_calc_splitting_index(start_index, end_index, split_axis);
        const int internal_node_index = cur_node_index;

        ArenaMeshBvhNode& internal_node = out_nodes[static_cast<size_t>(cur_node_index)];
        QuantizeArenaMeshBvhPoint(quantization, quantization.bvh_aabb_max, 0, internal_node.quantized_aabb_min);
        QuantizeArenaMeshBvhPoint(quantization, quantization.bvh_aabb_min, 1, internal_node.quantized_aabb_max);
        for (int i = start_index; i < end_index; ++i) {
            const ArenaMeshBvhNode& leaf = leaf_nodes[static_cast<size_t>(i)];
            uint16_t leaf_min[3];
            uint16_t leaf_max[3];
            QuantizeArenaMeshBvhPoint(quantization, GetArenaMeshBvhNodeAabbMin(quantization, leaf), 0, leaf_min);
            QuantizeArenaMeshBvhPoint(quantization, GetArenaMeshBvhNodeAabbMax(quantization, leaf), 1, leaf_max);
            for (int axis = 0; axis < 3; ++axis) {
                if (internal_node.quantized_aabb_min[axis] > leaf_min[axis]) {
                    internal_node.quantized_aabb_min[axis] = leaf_min[axis];
                }
                if (internal_node.quantized_aabb_max[axis] < leaf_max[axis]) {
                    internal_node.quantized_aabb_max[axis] = leaf_max[axis];
                }
            }
        }
        ++cur_node_index;

        const int left_child_node_index = cur_node_index;
        self(self, start_index, split_index);
        const int right_child_node_index = cur_node_index;
        self(self, split_index, end_index);

        const int escape_index = cur_node_index - cur_index;
        if (escape_index * static_cast<int>(sizeof(ArenaMeshBvhNode)) > kArenaMeshMaxSubtreeSizeBytes) {
            update_subtree_headers(left_child_node_index, right_child_node_index);
        }
        out_nodes[static_cast<size_t>(internal_node_index)].escape_index_or_triangle_index = -escape_index;
    };

    build_tree(build_tree, 0, mesh_triangle_count);
    out_nodes.resize(static_cast<size_t>(cur_node_index));

    if (out_subtrees.empty() && !out_nodes.empty()) {
        ArenaMeshBvhSubtreeInfo subtree;
        std::copy_n(out_nodes[0].quantized_aabb_min, 3, subtree.quantized_aabb_min);
        std::copy_n(out_nodes[0].quantized_aabb_max, 3, subtree.quantized_aabb_max);
        subtree.root_node_index = 0;
        subtree.subtree_size = out_nodes[0].is_leaf_node() ? 1 : out_nodes[0].get_escape_index();
        out_subtrees.push_back(subtree);
    }
}
}

std::vector<ArenaMeshTriangle> LoadArenaMeshTriangles(
    std::vector<ArenaMeshBvhQuantization>* out_quantization,
    std::vector<ArenaMeshBvhMeshInfo>* out_mesh_infos,
    std::vector<ArenaMeshBvhNode>* out_nodes,
    std::vector<ArenaMeshBvhSubtreeInfo>* out_subtrees,
    std::vector<ArenaMeshTriangle>* out_local_triangles,
    int* out_local_goal_meshes) {
    std::vector<ArenaMeshTriangle> triangles;
    const std::filesystem::path soccar_dir = OfficialCollisionMeshRoot() / "soccar";
    if (!std::filesystem::exists(soccar_dir)) {
        throw std::runtime_error("official RocketSim collision meshes not found");
    }

    int local_goal_meshes = 0;

    if (out_quantization) {
        out_quantization->assign(16, ArenaMeshBvhQuantization{});
    }
    if (out_mesh_infos) {
        out_mesh_infos->clear();
        out_mesh_infos->reserve(16);
    }
    if (out_nodes) {
        out_nodes->clear();
    }
    if (out_subtrees) {
        out_subtrees->clear();
    }

    for (int mesh_idx = 0; mesh_idx < 16; ++mesh_idx) {
        const std::filesystem::path mesh_path = soccar_dir / ("mesh_" + std::to_string(mesh_idx) + ".cmf");
        std::ifstream input(mesh_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("failed to open collision mesh: " + mesh_path.string());
        }

        int32_t num_tris = 0;
        int32_t num_vertices = 0;
        input.read(reinterpret_cast<char*>(&num_tris), sizeof(num_tris));
        input.read(reinterpret_cast<char*>(&num_vertices), sizeof(num_vertices));
        if (!input || num_tris <= 0 || num_vertices <= 0) {
            throw std::runtime_error("invalid collision mesh header: " + mesh_path.string());
        }

        std::vector<TriangleIndex> tri_indices(static_cast<size_t>(num_tris));
        std::vector<MeshVertex> vertices(static_cast<size_t>(num_vertices));
        input.read(reinterpret_cast<char*>(tri_indices.data()), static_cast<std::streamsize>(tri_indices.size() * sizeof(TriangleIndex)));
        input.read(reinterpret_cast<char*>(vertices.data()), static_cast<std::streamsize>(vertices.size() * sizeof(MeshVertex)));
        if (!input) {
            throw std::runtime_error("failed to read collision mesh: " + mesh_path.string());
        }

        const size_t mesh_base = triangles.size();
        Vec3 mesh_min(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        Vec3 mesh_max(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
        for (int32_t tri_i = 0; tri_i < num_tris; ++tri_i) {
            const TriangleIndex& tri_idx = tri_indices[static_cast<size_t>(tri_i)];
            if (tri_idx.vertex_index[0] < 0 || tri_idx.vertex_index[0] >= num_vertices ||
                tri_idx.vertex_index[1] < 0 || tri_idx.vertex_index[1] >= num_vertices ||
                tri_idx.vertex_index[2] < 0 || tri_idx.vertex_index[2] >= num_vertices) {
                throw std::runtime_error("collision mesh triangle index out of range: " + mesh_path.string());
            }

            const MeshVertex& va = vertices[static_cast<size_t>(tri_idx.vertex_index[0])];
            const MeshVertex& vb = vertices[static_cast<size_t>(tri_idx.vertex_index[1])];
            const MeshVertex& vc = vertices[static_cast<size_t>(tri_idx.vertex_index[2])];

            ArenaMeshTriangle tri;
            tri.a = Vec3(va.x, va.y, va.z);
            tri.b = Vec3(vb.x, vb.y, vb.z);
            tri.c = Vec3(vc.x, vc.y, vc.z);
            const Vec3 ab(tri.b.x - tri.a.x, tri.b.y - tri.a.y, tri.b.z - tri.a.z);
            const Vec3 ac(tri.c.x - tri.a.x, tri.c.y - tri.a.y, tri.c.z - tri.a.z);
            tri.normal = Normalize(Cross(ab, ac), Vec3(0.0f, 0.0f, 1.0f));
            tri.min = Vec3(
                Min(tri.a.x, Min(tri.b.x, tri.c.x)),
                Min(tri.a.y, Min(tri.b.y, tri.c.y)),
                Min(tri.a.z, Min(tri.b.z, tri.c.z)));
            tri.max = Vec3(
                Max(tri.a.x, Max(tri.b.x, tri.c.x)),
                Max(tri.a.y, Max(tri.b.y, tri.c.y)),
                Max(tri.a.z, Max(tri.b.z, tri.c.z)));
            tri.mesh_index = static_cast<uint8_t>(mesh_idx);
            mesh_min = MinComponents(mesh_min, tri.min);
            mesh_max = MaxComponents(mesh_max, tri.max);
            triangles.push_back(tri);
        }

        const float center_y = 0.5f * (mesh_min.y + mesh_max.y);
        const int8_t origin_sign = center_y > kGoalMeshOriginYBt ? 1 : (center_y < -kGoalMeshOriginYBt ? -1 : 0);

        for (int32_t tri_a = 0; tri_a < num_tris; ++tri_a) {
            const ArenaMeshTriangle& tri_a_ref = triangles[mesh_base + static_cast<size_t>(tri_a)];
            for (int32_t tri_b = 0; tri_b < num_tris; ++tri_b) {
                if (tri_a == tri_b) continue;
                const ArenaMeshTriangle& tri_b_ref = triangles[mesh_base + static_cast<size_t>(tri_b)];

                if (tri_a_ref.max.x < tri_b_ref.min.x || tri_a_ref.min.x > tri_b_ref.max.x ||
                    tri_a_ref.max.y < tri_b_ref.min.y || tri_a_ref.min.y > tri_b_ref.max.y ||
                    tri_a_ref.max.z < tri_b_ref.min.z || tri_a_ref.min.z > tri_b_ref.max.z) {
                    continue;
                }
                PopulateTriangleInternalEdgeInfo(
                    tri_indices,
                    triangles,
                    static_cast<int>(mesh_base + static_cast<size_t>(tri_a)),
                    static_cast<int>(mesh_base + static_cast<size_t>(tri_b)),
                    static_cast<int>(mesh_base));
            }
        }
        for (int32_t tri_i = 0; tri_i < num_tris; ++tri_i) {
            ArenaMeshTriangle& wt = triangles[mesh_base + static_cast<size_t>(tri_i)];
            if (wt.edge_v0v1_angle == 2.0f * Pi()) wt.edge_v0v1_angle = 0.0f;
            if (wt.edge_v1v2_angle == 2.0f * Pi()) wt.edge_v1v2_angle = 0.0f;
            if (wt.edge_v2v0_angle == 2.0f * Pi()) wt.edge_v2v0_angle = 0.0f;
        }

        if (out_local_triangles) {
            static constexpr float kGoalMeshLocalY[536] = {
                -0.208231956f, -0.805375218f, -0.805375218f, -6.25633335f, -8.17787647f, -8.1780262f, -6.2563324f, -1.75011611f,
                -3.00114131f, -3.00113893f, -1.75011122f, -3.00113893f, -1.75011122f, -0.562714756f, -0.328159034f, -0.329575032f,
                -0.564282119f, -1.53333008f, -1.60904539f, -1.23098147f, -1.17305171f, -0.888754785f, -0.84693104f, -1.36972642f,
                -1.41155016f, -0.937856257f, -0.91006577f, 0.0000021114231f, -0.00000032854885f, 0.00000982247184f, 0.0000122624715f, 0.0000139307685f,
                0.00001309662f, 0.0000106566195f, 15.4826374f, 15.4826374f, 16.1931648f, 16.1931553f, 17.222374f, 17.2223835f,
                17.5022964f, 17.5022964f, 17.2223721f, 17.2223816f, 16.7795315f, 16.7795258f, 16.1931438f, 16.1931438f,
                15.4826164f, 15.4826069f, 14.6673088f, 14.6673069f, 13.7666206f, 13.7666235f, -1.9550879f, -1.9550879f,
                -1.41155016f, -1.41155016f, -0.937856257f, -0.937856257f, -0.546928465f, -0.546928465f, -0.937856257f, -0.546928465f,
                -0.251693994f, -0.251693994f, -0.0650777072f, -0.0650777072f, -0.251693994f, -0.251693994f, 0.00000333662047f, 0.00000333662047f,
                0.00000250247194f, 0.00000661076911f, 0.00000166832331f, 0.00000166832331f, 0.000000521345953f, 0.000000521345953f, 0.00000661076911f, 0.00000988491593f,
                0.00000166832353f, 0.00000166832331f, 0.000000521345953f, 0.00000250247194f, 0.00000250247194f, 0.00000333662047f, -15.6772575f, -0.208236843f,
                -0.208236843f, -0.805380106f, -0.805380106f, 0.00000250247194f, 0.00000250247194f, 0.00000166832331f, 0.00000333662047f, 0.000011490768f,
                0.00000661076911f, 0.00000577662013f, 0.0000123249156f, -1.75011361f, -0.805380106f, -0.805377662f, -3.00114131f, 0.00000250247194f,
                0.00000166832331f, 0.00000333662047f, 0.160002083f, 0.159999639f, 14.6673088f, 15.4826069f, 4.27382517f, 13.9150963f,
                14.4025183f, 13.3792744f, 15.1039906f, 15.3599157f, 15.8032331f, 15.5033703f, 16.0746689f, 16.3112221f,
                16.7017288f, 16.4387417f, 17.1792393f, 17.3536339f, 17.448164f, 17.2701378f, 16.7017479f, 16.9187698f,
                16.5064449f, 16.3112297f, 16.074688f, 15.7934761f, 15.5916014f, -1.75011122f, -3.0011363f, -4.51703453f,
                -4.51703453f, 14.207757f, 14.335536f, 15.0061998f, 14.8227329f, -0.2082344f, -15.6778049f, 0.0000147649152f,
                0.00000982247184f, 0.0000147649152f, 0.000011490768f, 0.0000106566195f, 0.000011490768f, 0.0000147649152f, -0.208231956f, 12.7991867f,
                12.7993507f, 4.69311523f, 4.69306183f, -0.805375218f, 12.7994995f, 4.69316673f, 12.799634f, 4.69321012f,
                -4.51703453f, 12.7997484f, 4.69324684f, -6.25633335f, 12.799839f, 4.69327879f, -8.17766666f, 12.7999048f,
                4.69330072f, -15.6782637f, 0.00000494247115f, 12.7999392f, 4.69331026f, 0.00000821661979f, 12.7999458f, 4.69331551f,
                0.00000988491593f, 13.4572926f, -0.208236843f, -0.2082344f, 13.5299826f, 13.5957832f, -4.5170393f, 13.6531219f,
                -6.2563386f, -6.25633574f, -8.17782784f, -6.25634098f, -8.17779541f, -8.17773628f, 13.7004414f, -15.6780682f,
                -15.6778736f, 0.00000050589864f, 0.00000898832332f, 0.000000467280472f, 13.7361736f, 0.000000417076791f, 0.000000366873167f, 13.7587566f,
                0.00000032825497f, 0.000000312807714f, 13.7666235f, -0.000000328576505f, -0.000000335327002f, 0.00000738247172f, 13.9151726f, 14.0666571f,
                -0.00000279510436f, -0.00000526638405f, 14.2077999f, -0.0000298991927f, -0.0000372861068f, 14.3355589f, -0.00002251554f, -0.0000151179484f,
                14.4468966f, -0.0000101877813f, 14.5387812f, -0.00121062726f, -0.0402829498f, -0.03904761f, 14.6081686f, -0.152324021f,
                -0.15101786f, 14.6520233f, 14.6673088f, -0.848676682f, 14.4025612f, 14.6200848f, -1.17500484f, 14.8227587f,
                -1.5355078f, 15.0062151f, -2.94249988f, -2.93945312f, 15.1660957f, 15.2980394f, 15.3976755f, -3.08457994f,
                -0.590502799f, 15.4606543f, -0.344362646f, 15.4826069f, -0.158473924f, 14.8293724f, 15.1040354f, -0.040973857f,
                15.3599482f, -1.79834962f, -3.44749022f, -1.37580323f, 15.9600782f, -0.993312895f, 16.0858974f, -0.659973025f,
                16.1654186f, -0.384875327f, 16.1931438f, -0.177118912f, 15.1815691f, 15.5033979f, -0.0457956083f, 15.8032513f,
                -2.04443359f, -3.91926742f, -1.56406975f, 16.6538677f, -1.12924063f, 16.7470398f, -0.750285447f, 16.7795315f,
                -0.43754375f, 15.4458504f, 15.8036432f, -0.201357171f, 16.1370106f, -0.0520602055f, 16.4387684f, -2.29052734f,
                17.0826664f, -4.39100599f, -1.75233889f, 17.1862602f, -1.26516831f, 17.2223721f, -0.840595543f, 15.6119537f,
                15.9926767f, -0.490212202f, 16.347393f, -0.225592971f, 16.6684875f, -0.0583272688f, 16.9483109f, 17.1792393f,
                -2.47982407f, 17.3536339f, -4.7539258f, -1.89715815f, 17.4638672f, 17.5022964f, 15.6702356f, 16.0588779f,
                -0.530724883f, 16.4209766f, -0.244237974f, 16.7487507f, -0.0631490126f, 17.0343952f, -2.55554676f, -4.89906216f,
                17.5606842f, 17.5999126f, -0.546928465f, 15.6119537f, 15.9926672f, -0.251693994f, 16.347393f, -0.0650777072f,
                16.6684875f, 0.000000521345953f, 16.9483109f, -0.00000032854885f, 17.4638672f, 17.5022964f, 4.69331789f, 4.69331551f,
                12.7999458f, 12.7999439f, 15.4458017f, 15.8036041f, 14.6673613f, 4.48001194f, 4.48001194f, 14.6673641f,
                16.1369743f, 16.7795315f, 16.7795315f, 16.9187603f, 17.5999088f, 17.5999126f, 17.502306f, 17.5022964f,
                17.5999088f, 17.5022964f, 17.2223721f, 17.0826664f, 17.1862602f, 17.222374f, 15.1815319f, 0.000000521345953f,
                -2.55554676f, -4.89906216f, 16.5064373f, 16.6538677f, -0.251693994f, 16.7470512f, -0.0650777072f, 16.7795315f,
                0.000000521345953f, 14.8293104f, -2.55554676f, 15.5915728f, -4.89906216f, -1.9550879f, 15.7934675f, -2.55554676f,
                -1.9550879f, 15.9600792f, 16.0858994f, -0.0650777072f, 16.1654301f, 16.1931553f, -4.89906216f, 14.6200523f,
                -1.41155016f, -0.0650777072f, 15.1660919f, 15.2980452f, -2.55554676f, 15.3976965f, -4.89906216f, -1.9550879f,
                15.4606743f, -1.41155016f, 15.4826374f, -0.937856257f, 14.0665998f, -0.546928465f, 14.4468927f, 14.5387926f,
                14.6081944f, -4.89906216f, -2.55554676f, 14.6520643f, -1.9550879f, 14.6673613f, -1.41155016f, 4.31534863f,
                -0.937856257f, 4.35403728f, -0.546928465f, 4.38905668f, 4.41957951f, 4.44476748f, 4.69331551f, 12.7999458f,
                4.46379089f, 4.47581768f, 4.48001194f, 0.159972534f, 0.00000988491593f, 0.159965143f, -0.208236843f, 0.15997991f,
                -0.805380106f, 0.159984857f, -1.75011849f, -1.75011849f, 0.159989789f, -3.00114608f, -3.00114608f, 0.159994707f,
                -4.5170393f, -4.5170393f, 0.15999718f, -6.25634098f, -6.25634098f, 0.159999639f, -8.1776886f, -8.1776886f,
                0.159999639f, -15.6773949f, -15.6773949f, 0.00000661076911f, 0.00000988491593f, -1.75011849f, -3.00114608f, 13.7666235f,
                -4.5170393f, -6.25634098f, -8.1776886f, -15.6773853f, 16.1931438f, 16.7795258f, -0.208236843f, 0.00000988491593f,
                -0.805380106f, -1.75011849f, 17.5022964f, -3.00114608f, 17.222374f, -4.5170393f, 16.7795315f, -6.25634098f,
                16.1931553f, -8.1776886f, 15.4826374f, -15.6782341f, 14.6673613f, 4.48001194f, 0.00000333662047f, 0.00000661076911f,
                0.00000661076911f, 0.159999639f, 0.00000988491593f, 0.00000661076911f, -0.208236843f, 0.159983158f, 0.160000294f, -0.805380106f,
                0.159934267f, -1.75011849f, 0.159856081f, -3.00114608f, 0.159751073f, 0.159626514f, -6.25634098f, -4.5170393f,
                0.159482449f, -8.177845f, 0.159318849f, -15.6779127f, 0.159147933f, -4.51703453f, 0.00000988491593f, -0.208236843f,
                -0.805380106f, -1.75011849f, -3.00114608f, -4.5170393f, -15.6780195f, -0.127153128f, -0.291064322f, 0.0385963917f,
                -0.0307982005f, -0.186208352f, -0.0603782274f, 0.0191579703f, -0.550788462f, -0.181618646f, 0.158671856f, -0.508102953f,
                -0.465351552f, -0.830627322f, -0.108920865f, -1.15172362f, -0.0431151874f, -1.50644529f, 0.0142334914f, -2.89086914f,
                -1.05371094f, -0.502910137f, -0.771093667f, -0.361774862f, -1.07286131f, -0.234020934f, -1.40623045f, -0.122680567f,
                -2.70732427f, -0.441032618f, -0.352819741f, -0.677604914f, -1.58313477f, -0.792053223f, -0.949055135f, -0.589370072f,
                -1.24893546f, -0.405895948f, -2.41927719f, -0.245998427f, -0.554709435f, -2.04109359f, -1.04227531f, -0.786364734f,
            };
            static constexpr uint16_t kGoalMeshMirrorSwap[51] = {
                4, 8, 14, 18, 24, 28, 32, 35, 39, 43, 47, 51, 55, 57, 62, 67, 71,
                81, 88, 92, 96, 103, 106, 111, 115, 119, 123, 127, 138, 152, 156, 158, 161, 164,
                167, 171, 174, 185, 187, 194, 203, 220, 223, 234, 327, 341, 367, 393, 446, 478, 493,
            };
            std::vector<MeshVertex> frame_vertices = vertices;
            if (origin_sign != 0 && frame_vertices.size() == 536) {
                for (size_t i = 0; i < 536; ++i) {
                    frame_vertices[i].y = static_cast<float>(origin_sign) * kGoalMeshLocalY[i];
                    if (frame_vertices[i].x == 0.0f && (mesh_idx == 4 || mesh_idx == 7))
                        frame_vertices[i].x = -0.0f;
                }
                if ((mesh_idx & 1) != 0) {
                    for (uint16_t s : kGoalMeshMirrorSwap) {
                        std::swap(frame_vertices[s].y, frame_vertices[s + 1].y);
                    }
                }
            } else {
                for (MeshVertex& v : frame_vertices) v.y -= static_cast<float>(origin_sign) * kGoalMeshOriginYBt;
            }
            local_goal_meshes += origin_sign != 0;

            std::vector<ArenaMeshTriangle>& local = *out_local_triangles;
            for (int32_t tri_i = 0; tri_i < num_tris; ++tri_i) {
                const TriangleIndex& tri_idx = tri_indices[static_cast<size_t>(tri_i)];
                const MeshVertex& va = frame_vertices[static_cast<size_t>(tri_idx.vertex_index[0])];
                const MeshVertex& vb = frame_vertices[static_cast<size_t>(tri_idx.vertex_index[1])];
                const MeshVertex& vc = frame_vertices[static_cast<size_t>(tri_idx.vertex_index[2])];
                ArenaMeshTriangle tri = triangles[mesh_base + static_cast<size_t>(tri_i)];
                tri.a = Vec3(va.x, va.y, va.z);
                tri.b = Vec3(vb.x, vb.y, vb.z);
                tri.c = Vec3(vc.x, vc.y, vc.z);
                const Vec3 ab(tri.b.x - tri.a.x, tri.b.y - tri.a.y, tri.b.z - tri.a.z);
                const Vec3 ac(tri.c.x - tri.a.x, tri.c.y - tri.a.y, tri.c.z - tri.a.z);
                tri.normal = Normalize(Cross(ab, ac), Vec3(0.0f, 0.0f, 1.0f));
                tri.min = Vec3(
                    Min(tri.a.x, Min(tri.b.x, tri.c.x)),
                    Min(tri.a.y, Min(tri.b.y, tri.c.y)),
                    Min(tri.a.z, Min(tri.b.z, tri.c.z)));
                tri.max = Vec3(
                    Max(tri.a.x, Max(tri.b.x, tri.c.x)),
                    Max(tri.a.y, Max(tri.b.y, tri.c.y)),
                    Max(tri.a.z, Max(tri.b.z, tri.c.z)));
                tri.edge_v0v1_angle = 2.0f * Pi();
                tri.edge_v1v2_angle = 2.0f * Pi();
                tri.edge_v2v0_angle = 2.0f * Pi();
                tri.edge_flags = 0;
                tri.cal_origin_sign = origin_sign;
                local.push_back(tri);
            }
            for (int32_t tri_a = 0; tri_a < num_tris; ++tri_a) {
                const ArenaMeshTriangle& tri_a_ref = local[mesh_base + static_cast<size_t>(tri_a)];
                for (int32_t tri_b = 0; tri_b < num_tris; ++tri_b) {
                    if (tri_a == tri_b) continue;
                    const ArenaMeshTriangle& tri_b_ref = local[mesh_base + static_cast<size_t>(tri_b)];
                    if (tri_a_ref.max.x < tri_b_ref.min.x || tri_a_ref.min.x > tri_b_ref.max.x ||
                        tri_a_ref.max.y < tri_b_ref.min.y || tri_a_ref.min.y > tri_b_ref.max.y ||
                        tri_a_ref.max.z < tri_b_ref.min.z || tri_a_ref.min.z > tri_b_ref.max.z) {
                        continue;
                    }
                    PopulateTriangleInternalEdgeInfo(
                        tri_indices,
                        local,
                        static_cast<int>(mesh_base + static_cast<size_t>(tri_a)),
                        static_cast<int>(mesh_base + static_cast<size_t>(tri_b)),
                        static_cast<int>(mesh_base));
                }
            }

            for (int32_t tri_i = 0; tri_i < num_tris; ++tri_i) {
                ArenaMeshTriangle& lt = local[mesh_base + static_cast<size_t>(tri_i)];
                if (lt.edge_v0v1_angle == 2.0f * Pi()) lt.edge_v0v1_angle = 0.0f;
                if (lt.edge_v1v2_angle == 2.0f * Pi()) lt.edge_v1v2_angle = 0.0f;
                if (lt.edge_v2v0_angle == 2.0f * Pi()) lt.edge_v2v0_angle = 0.0f;
            }
        }

        if (out_quantization) {
            const bool local_bvh = out_local_triangles != nullptr && origin_sign != 0;
            std::vector<ArenaMeshTriangle>& bvh_tris = local_bvh ? *out_local_triangles : triangles;
            Vec3 bvh_min = mesh_min;
            Vec3 bvh_max = mesh_max;
            if (local_bvh) {
                bvh_min = bvh_tris[mesh_base].min;
                bvh_max = bvh_tris[mesh_base].max;
                for (int32_t tri_i = 1; tri_i < num_tris; ++tri_i) {
                    bvh_min = MinComponents(bvh_min, bvh_tris[mesh_base + static_cast<size_t>(tri_i)].min);
                    bvh_max = MaxComponents(bvh_max, bvh_tris[mesh_base + static_cast<size_t>(tri_i)].max);
                }
            }
            SetArenaMeshBvhQuantization((*out_quantization)[static_cast<size_t>(mesh_idx)], bvh_min, bvh_max, 1.0f, out_local_triangles != nullptr);
            const ArenaMeshBvhQuantization& quantization = (*out_quantization)[static_cast<size_t>(mesh_idx)];
            for (size_t tri_pos = mesh_base; tri_pos < triangles.size(); ++tri_pos) {
                ArenaMeshTriangle& tri = triangles[tri_pos];
                const ArenaMeshTriangle& src = bvh_tris[tri_pos];
                QuantizeArenaMeshBvhPoint(quantization, src.min, 0, tri.quantized_aabb_min);
                QuantizeArenaMeshBvhPoint(quantization, src.max, 1, tri.quantized_aabb_max);
                if (out_local_triangles) {
                    std::copy_n(tri.quantized_aabb_min, 3, (*out_local_triangles)[tri_pos].quantized_aabb_min);
                    std::copy_n(tri.quantized_aabb_max, 3, (*out_local_triangles)[tri_pos].quantized_aabb_max);
                }
            }

            if (out_mesh_infos && out_nodes && out_subtrees) {
                std::vector<ArenaMeshBvhNode> mesh_nodes;
                std::vector<ArenaMeshBvhSubtreeInfo> mesh_subtrees;
                BuildArenaMeshBulletStyleBvh(
                    bvh_tris,
                    static_cast<int32_t>(mesh_base),
                    num_tris,
                    quantization,
                    mesh_nodes,
                    mesh_subtrees);

                ArenaMeshBvhMeshInfo mesh_info;
                mesh_info.aabb_min = mesh_min;
                mesh_info.aabb_max = mesh_max;
                mesh_info.mesh_index = mesh_idx;
                mesh_info.cal_origin_y = static_cast<float>(origin_sign) * kGoalMeshOriginYBt;
                mesh_info.triangle_start = static_cast<int32_t>(mesh_base);
                mesh_info.triangle_count = num_tris;
                mesh_info.node_offset = static_cast<int32_t>(out_nodes->size());
                mesh_info.node_count = static_cast<int32_t>(mesh_nodes.size());
                mesh_info.subtree_offset = static_cast<int32_t>(out_subtrees->size());
                mesh_info.subtree_count = static_cast<int32_t>(mesh_subtrees.size());
                mesh_info.dfs_subtree_offset = mesh_info.subtree_offset + mesh_info.subtree_count;
                out_mesh_infos->push_back(mesh_info);

                out_nodes->insert(out_nodes->end(), mesh_nodes.begin(), mesh_nodes.end());
                for (ArenaMeshBvhSubtreeInfo& subtree : mesh_subtrees) {
                    subtree.root_node_index += mesh_info.node_offset;
                }
                out_subtrees->insert(out_subtrees->end(), mesh_subtrees.begin(), mesh_subtrees.end());
                std::stable_sort(mesh_subtrees.begin(), mesh_subtrees.end(),
                                 [](const ArenaMeshBvhSubtreeInfo& a, const ArenaMeshBvhSubtreeInfo& b) {
                                     return a.root_node_index < b.root_node_index;
                                 });
                out_subtrees->insert(out_subtrees->end(), mesh_subtrees.begin(), mesh_subtrees.end());
            }
        }
    }
    if (out_local_goal_meshes) *out_local_goal_meshes = local_goal_meshes;
    return triangles;
}

std::vector<uint16_t> BuildArenaBroadphaseCellMask(
    const std::vector<ArenaMeshTriangle>& triangles,
    const std::vector<ArenaMeshBvhMeshInfo>& mesh_infos,
    ArenaBroadphaseGrid& out_grid) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    out_grid.min_bt = Vec3(-5600.0f * kUUToBT, -6000.0f * kUUToBT, 0.0f);
    out_grid.max_bt = Vec3(5600.0f * kUUToBT, 6000.0f * kUUToBT, 2200.0f * kUUToBT);
    out_grid.cell_size_bt = 370.0f * kUUToBT;
    const Vec3 range(
        out_grid.max_bt.x - out_grid.min_bt.x,
        out_grid.max_bt.y - out_grid.min_bt.y,
        out_grid.max_bt.z - out_grid.min_bt.z);
    out_grid.cells_x = std::max(1, static_cast<int32_t>(std::ceil(range.x / out_grid.cell_size_bt)));
    out_grid.cells_y = std::max(1, static_cast<int32_t>(std::ceil(range.y / out_grid.cell_size_bt)));
    out_grid.cells_z = std::max(1, static_cast<int32_t>(std::ceil(range.z / out_grid.cell_size_bt)));
    out_grid.cell_count = out_grid.cells_x * out_grid.cells_y * out_grid.cells_z;

    std::vector<uint16_t> mask(static_cast<size_t>(out_grid.cell_count), 0);
    if (triangles.empty() || mesh_infos.empty()) {
        out_grid.cell_mesh_mask = nullptr;
        return mask;
    }

    const auto cell_indices = [&out_grid](const Vec3& p, int& i, int& j, int& k) {
        i = static_cast<int>((p.x - out_grid.min_bt.x) / out_grid.cell_size_bt);
        j = static_cast<int>((p.y - out_grid.min_bt.y) / out_grid.cell_size_bt);
        k = static_cast<int>((p.z - out_grid.min_bt.z) / out_grid.cell_size_bt);
        i = std::min(std::max(i, 0), out_grid.cells_x - 1);
        j = std::min(std::max(j, 0), out_grid.cells_y - 1);
        k = std::min(std::max(k, 0), out_grid.cells_z - 1);
    };

    for (const ArenaMeshBvhMeshInfo& info : mesh_infos) {
        if (info.mesh_index < 0 || info.mesh_index >= 16 || info.triangle_count <= 0) {
            continue;
        }
        const Vec3 clamped_max(
            std::min(info.aabb_max.x, out_grid.max_bt.x),
            std::min(info.aabb_max.y, out_grid.max_bt.y),
            std::min(info.aabb_max.z, out_grid.max_bt.z));
        int i_min = 0, j_min = 0, k_min = 0, i_max = 0, j_max = 0, k_max = 0;
        cell_indices(info.aabb_min, i_min, j_min, k_min);
        cell_indices(clamped_max, i_max, j_max, k_max);
        const uint16_t bit = static_cast<uint16_t>(1u << info.mesh_index);
        for (int i = i_min; i <= i_max; ++i) {
            for (int j = j_min; j <= j_max; ++j) {
                for (int k = k_min; k <= k_max; ++k) {
                    const Vec3 cell_min(
                        out_grid.min_bt.x + static_cast<float>(i) * out_grid.cell_size_bt,
                        out_grid.min_bt.y + static_cast<float>(j) * out_grid.cell_size_bt,
                        out_grid.min_bt.z + static_cast<float>(k) * out_grid.cell_size_bt);
                    const Vec3 cell_max(
                        cell_min.x + out_grid.cell_size_bt,
                        cell_min.y + out_grid.cell_size_bt,
                        cell_min.z + out_grid.cell_size_bt);
                    bool has_triangle = false;
                    const size_t begin = static_cast<size_t>(info.triangle_start);
                    const size_t end = begin + static_cast<size_t>(info.triangle_count);
                    for (size_t t = begin; t < end && t < triangles.size(); ++t) {
                        const ArenaMeshTriangle& tri = triangles[t];
                        if (tri.max.x < cell_min.x || tri.min.x > cell_max.x ||
                            tri.max.y < cell_min.y || tri.min.y > cell_max.y ||
                            tri.max.z < cell_min.z || tri.min.z > cell_max.z) {
                            continue;
                        }
                        has_triangle = true;
                        break;
                    }
                    if (!has_triangle) {
                        continue;
                    }
                    for (int di = -1; di <= 1; ++di) {
                        for (int dj = -1; dj <= 1; ++dj) {
                            for (int dk = -1; dk <= 1; ++dk) {
                                const int ci = i + di;
                                const int cj = j + dj;
                                const int ck = k + dk;
                                if (ci < 0 || cj < 0 || ck < 0 ||
                                    ci >= out_grid.cells_x || cj >= out_grid.cells_y || ck >= out_grid.cells_z) {
                                    continue;
                                }
                                mask[static_cast<size_t>(ci * out_grid.cells_y * out_grid.cells_z +
                                                         cj * out_grid.cells_z + ck)] |= bit;
                            }
                        }
                    }
                }
            }
        }
    }
    return mask;
}

void BuildArenaMeshClearance(
    const std::vector<ArenaMeshTriangle>& mesh,
    const std::vector<ArenaMeshTriangle>& local_mesh,
    std::vector<uint8_t>& out,
    Vec3& out_lo,
    float& out_inv_cell,
    float& out_unit,
    int32_t out_dims[3],
    int threads) {
    constexpr double kCell = 0.5;
    constexpr double kMax = 3.0;
    constexpr double kSlop = 0.01;
    out.clear();
    struct Tri {
        double v[3][3];
    };
    std::vector<Tri> tris;
    const auto add = [&](const ArenaMeshTriangle& t, double oy) {
        Tri r;
        const Vec3* p[3] = {&t.a, &t.b, &t.c};
        for (int k = 0; k < 3; ++k) {
            r.v[k][0] = p[k]->x;
            r.v[k][1] = static_cast<double>(p[k]->y) + oy;
            r.v[k][2] = p[k]->z;
        }
        tris.push_back(r);
    };
    for (size_t i = 0; i < mesh.size(); ++i) {
        add(mesh[i], 0.0);
        if (i < local_mesh.size() && local_mesh[i].cal_origin_sign != 0) {
            add(local_mesh[i], static_cast<double>(static_cast<float>(local_mesh[i].cal_origin_sign) * kGoalMeshOriginYBt));
        }
    }
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (const Tri& t : tris) {
        for (int k = 0; k < 3; ++k) {
            for (int a = 0; a < 3; ++a) {
                if (!std::isfinite(t.v[k][a]) || std::abs(t.v[k][a]) > 256.0) {
                    return;
                }
                lo[a] = std::min(lo[a], t.v[k][a]);
                hi[a] = std::max(hi[a], t.v[k][a]);
            }
        }
    }
    if (tris.empty()) {
        return;
    }
    int dims[3];
    for (int a = 0; a < 3; ++a) {
        lo[a] = std::floor((lo[a] - kMax) / kCell) * kCell;
        const double count = std::ceil((hi[a] + kMax - lo[a]) / kCell);
        if (!(count > 0.0 && count <= 4096.0)) {
            return;
        }
        dims[a] = static_cast<int>(count);
    }
    const double unit = kMax / 255.0;
    const double half_diag = 0.5 * kCell * std::sqrt(3.0);
    out.assign(static_cast<size_t>(dims[0]) * dims[1] * dims[2], 255);
    const auto sub = [](const double* a, const double* b, double* o) {
        o[0] = a[0] - b[0];
        o[1] = a[1] - b[1];
        o[2] = a[2] - b[2];
    };
    const auto dot = [](const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    const auto fill_slab = [&](int z_begin, int z_end) {
        for (const Tri& t : tris) {
            const double* a = t.v[0];
            const double* b = t.v[1];
            const double* c = t.v[2];
            double ab[3], ac[3], bc[3];
            sub(b, a, ab);
            sub(c, a, ac);
            sub(c, b, bc);
            const double n[3] = {
                ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]};
            const double n_len = std::sqrt(dot(n, n));
            const double max_edge = std::sqrt(std::max(dot(ab, ab), std::max(dot(ac, ac), dot(bc, bc))));
            const double tol =
                n_len > 0.0 ? max_edge * 5e-4 + (1e-5 * max_edge + 6e-5) * max_edge * max_edge / n_len : kMax;
            const double pad = kSlop + tol + half_diag;
            const double reach = kMax + tol;
            int i0[3], i1[3];
            for (int ax = 0; ax < 3; ++ax) {
                const double tmin = std::min(a[ax], std::min(b[ax], c[ax])) - reach;
                const double tmax = std::max(a[ax], std::max(b[ax], c[ax])) + reach;
                i0[ax] = std::max(0, static_cast<int>(std::floor((tmin - lo[ax]) / kCell - 0.5)));
                i1[ax] = std::min(dims[ax] - 1, static_cast<int>(std::ceil((tmax - lo[ax]) / kCell - 0.5)));
            }
            const double plane_reach = reach * n_len * (1.0 + 1e-9) + 1e-9;
            const double cen[3] = {(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0};
            double r2 = 0.0;
            for (const double* v : {a, b, c}) {
                double dv[3];
                sub(v, cen, dv);
                r2 = std::max(r2, dot(dv, dv));
            }
            const double sphere_reach = (std::sqrt(r2) + reach) * (1.0 + 1e-9) + 1e-9;
            const double cap_lb = kMax - pad;
            const int cap_code =
                !(cap_lb > 0.0) ? 0 : cap_lb >= kMax ? 255 : static_cast<int>(std::floor(cap_lb / unit));
            for (int iz = std::max(i0[2], z_begin); iz <= std::min(i1[2], z_end - 1); ++iz) {
                for (int iy = i0[1]; iy <= i1[1]; ++iy) {
                    uint8_t* const row = &out[(static_cast<size_t>(iz) * dims[1] + iy) * dims[0]];
                    for (int ix = i0[0]; ix <= i1[0]; ++ix) {
                        row[ix] = static_cast<uint8_t>(std::min<int>(row[ix], cap_code));
                    }
                    int x_begin = i0[0], x_end = i1[0];
                    {
                        const double dy = lo[1] + (iy + 0.5) * kCell - cen[1];
                        const double dz = lo[2] + (iz + 0.5) * kCell - cen[2];
                        const double rem = sphere_reach * sphere_reach - dy * dy - dz * dz;
                        if (rem < 0.0) {
                            continue;
                        }
                        const double hw = std::sqrt(rem);
                        const double sb0 = std::floor((cen[0] - hw - lo[0]) / kCell - 0.5) - 1.0;
                        const double sb1 = std::ceil((cen[0] + hw - lo[0]) / kCell - 0.5) + 1.0;
                        x_begin = std::max(x_begin, static_cast<int>(std::min(std::max(sb0, -1.0), 4097.0)));
                        x_end = std::min(x_end, static_cast<int>(std::min(std::max(sb1, -1.0), 4097.0)));
                    }
                    double k = 0.0;
                    if (n_len > 0.0) {
                        k = n[1] * (lo[1] + (iy + 0.5) * kCell - a[1]) + n[2] * (lo[2] + (iz + 0.5) * kCell - a[2]) -
                            n[0] * a[0];
                        if (std::abs(n[0]) > 1e-12 * n_len) {
                            const double xa = (-plane_reach - k) / n[0], xb = (plane_reach - k) / n[0];
                            const double x_lo = std::min(xa, xb), x_hi = std::max(xa, xb);
                            const double xb0 = std::floor((x_lo - lo[0]) / kCell - 0.5) - 1.0;
                            const double xb1 = std::ceil((x_hi - lo[0]) / kCell - 0.5) + 1.0;
                            x_begin = std::max(x_begin, static_cast<int>(std::min(std::max(xb0, -1.0), 4097.0)));
                            x_end = std::min(x_end, static_cast<int>(std::min(std::max(xb1, -1.0), 4097.0)));
                        } else if (std::abs(k) > plane_reach) {
                            continue;
                        }
                    }
                    for (int ix = x_begin; ix <= x_end; ++ix) {
                        const double p[3] = {
                            lo[0] + (ix + 0.5) * kCell, lo[1] + (iy + 0.5) * kCell, lo[2] + (iz + 0.5) * kCell};
                        if (n_len > 0.0) {
                            const double plane_lb = std::min(std::abs(n[0] * p[0] + k) / n_len - 1e-9, kMax) - pad;
                            const int plane_code = !(plane_lb > 0.0) ? 0
                                : plane_lb >= kMax ? 255 : static_cast<int>(std::floor(plane_lb / unit));
                            if (plane_code >= row[ix]) {
                                continue;
                            }
                        }
                        double ap[3], bp[3], cp[3], q[3];
                        sub(p, a, ap);
                        sub(p, b, bp);
                        sub(p, c, cp);
                        const double d1 = dot(ab, ap), d2 = dot(ac, ap), d3 = dot(ab, bp), d4 = dot(ac, bp);
                        const double d5 = dot(ab, cp), d6 = dot(ac, cp);
                        const double vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
                        double u = 0.0, v = 0.0;
                        if (d1 <= 0.0 && d2 <= 0.0) {
                        } else if (d3 >= 0.0 && d4 <= d3) {
                            u = 1.0;
                        } else if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
                            u = d1 / (d1 - d3);
                        } else if (d6 >= 0.0 && d5 <= d6) {
                            v = 1.0;
                        } else if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
                            v = d2 / (d2 - d6);
                        } else if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
                            const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            u = 1.0 - w;
                            v = w;
                        } else {
                            const double den = 1.0 / (va + vb + vc);
                            u = vb * den;
                            v = vc * den;
                        }
                        for (int k = 0; k < 3; ++k) {
                            q[k] = a[k] + ab[k] * u + ac[k] * v - p[k];
                        }
                        const double lb = std::min(std::sqrt(dot(q, q)), kMax) - pad;
                        const int code = !(lb > 0.0) ? 0 : lb >= kMax ? 255 : static_cast<int>(std::floor(lb / unit));
                        row[ix] = static_cast<uint8_t>(std::min<int>(row[ix], code));
                    }
                }
            }
        }
    };
    const int n_threads = std::max(1, std::min(threads, dims[2]));
    std::vector<std::thread> workers;
    try {
        for (int k = 1; k < n_threads; ++k) {
            workers.emplace_back(fill_slab, dims[2] * k / n_threads, dims[2] * (k + 1) / n_threads);
        }
        fill_slab(0, dims[2] / n_threads);
    } catch (...) {
        for (std::thread& w : workers) w.join();
        throw;
    }
    for (std::thread& w : workers) {
        w.join();
    }
    const int far_code = static_cast<int>(std::floor((kMax - kSlop - half_diag) / unit));
    for (uint8_t& cell : out) {
        cell = static_cast<uint8_t>(std::min<int>(cell, far_code));
    }
    out_lo = Vec3(static_cast<float>(lo[0]), static_cast<float>(lo[1]), static_cast<float>(lo[2]));
    out_inv_cell = static_cast<float>(1.0 / kCell);
    out_unit = static_cast<float>(unit);
    out_dims[0] = dims[0];
    out_dims[1] = dims[1];
    out_dims[2] = dims[2];
}

void SetArenaMeshFreeBox(std::vector<ArenaMeshBvhMeshInfo>& mesh_infos) {
    const auto gap = [](float lo, float hi) { return lo > 0.0f ? lo : (hi < 0.0f ? -hi : 0.0f); };
    constexpr float kUnbounded = 1.0e4f;
    float best_x = 0.0f, best_y = 0.0f;
    std::vector<float> candidates(1, kUnbounded);
    for (const ArenaMeshBvhMeshInfo& m : mesh_infos) {
        candidates.push_back(gap(m.aabb_min.x, m.aabb_max.x));
    }
    for (const float x : candidates) {
        if (x <= 0.0f) {
            continue;
        }
        float y = kUnbounded;
        for (const ArenaMeshBvhMeshInfo& m : mesh_infos) {
            if (gap(m.aabb_min.x, m.aabb_max.x) < x) {
                y = std::min(y, gap(m.aabb_min.y, m.aabb_max.y));
            }
        }
        if (y > 0.0f && x * y > best_x * best_y) {
            best_x = x;
            best_y = y;
        }
    }
    for (ArenaMeshBvhMeshInfo& m : mesh_infos) {
        m.free_half_x = best_x;
        m.free_half_y = best_y;
    }
}

void SetArenaMeshRayOrder(std::vector<ArenaMeshBvhMeshInfo>& mesh_infos, float neg_x_plane_max_x) {
    const auto key = [&](size_t i) { return ArenaSweepQuantize(mesh_infos[i].aabb_max.x, 1); };
    std::vector<int32_t> order(mesh_infos.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = static_cast<int32_t>(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return key(a) < key(b); });
    const uint16_t plane_key = ArenaSweepQuantize(neg_x_plane_max_x, 1);
    int32_t plane_pos = 0;
    for (size_t i = 0; i < mesh_infos.size(); ++i) {
        plane_pos += key(i) <= plane_key ? 1 : 0;
    }
    for (size_t k = 0; k < order.size(); ++k) {
        mesh_infos[k].ray_order = order[k];
        mesh_infos[k].ray_neg_x_plane_pos = plane_pos;
    }
}

void BuildArenaMeshLeafGrid(
    const std::vector<ArenaMeshBvhQuantization>& quantization,
    const std::vector<ArenaMeshBvhMeshInfo>& mesh_infos,
    const std::vector<ArenaMeshBvhNode>& nodes,
    const std::vector<ArenaMeshBvhSubtreeInfo>& subtrees,
    std::vector<ArenaMeshLeafGridInfo>& out_infos,
    std::vector<int32_t>& out_cell_start,
    std::vector<ArenaMeshLeafEntry>& out_entries,
    float cell_bt) {
    const float kCellBt = cell_bt;
    const auto contains = [](const uint16_t* outer_min, const uint16_t* outer_max,
                             const uint16_t* inner_min, const uint16_t* inner_max) {
        for (int a = 0; a < 3; ++a) {
            if (inner_min[a] < outer_min[a] || inner_max[a] > outer_max[a]) {
                return false;
            }
        }
        return true;
    };
    out_infos.assign(mesh_infos.size(), ArenaMeshLeafGridInfo{});
    out_cell_start.assign(1, 0);
    out_entries.clear();
    for (size_t m = 0; m < mesh_infos.size(); ++m) {
        const ArenaMeshBvhMeshInfo& mesh = mesh_infos[m];
        const int begin = mesh.node_offset;
        const int end = mesh.node_offset + mesh.node_count;
        if (mesh.mesh_index < 0 || mesh.mesh_index >= static_cast<int>(quantization.size()) ||
            mesh.node_count <= 0 || begin < 0 || end > static_cast<int>(nodes.size())) {
            continue;
        }
        bool nested = true;
        for (int i = begin; nested && i < end; ++i) {
            if (nodes[i].is_leaf_node()) {
                continue;
            }
            const int stop = std::min(end, i + nodes[i].get_escape_index());
            for (int j = i + 1; nested && j < stop; ++j) {
                nested = contains(nodes[i].quantized_aabb_min, nodes[i].quantized_aabb_max,
                                  nodes[j].quantized_aabb_min, nodes[j].quantized_aabb_max);
            }
        }
        const bool headers = mesh.subtree_count > 0 && mesh.subtree_offset >= 0 && mesh.dfs_subtree_offset >= 0 &&
                             mesh.dfs_subtree_offset + mesh.subtree_count <= static_cast<int>(subtrees.size());
        if (nested && headers) {
            std::vector<uint8_t> covered(static_cast<size_t>(mesh.node_count), 0);
            int last_leaf = begin - 1;
            for (int h = 0; nested && h < mesh.subtree_count; ++h) {
                const ArenaMeshBvhSubtreeInfo& header = subtrees[static_cast<size_t>(mesh.dfs_subtree_offset + h)];
                const int root = header.root_node_index;
                if (root < begin || root >= end || header.subtree_size <= 0 || header.subtree_size > end - root) {
                    nested = false;
                    break;
                }
                const int stop = root + header.subtree_size;
                for (int j = root; nested && j < stop; ++j) {
                    if (nodes[j].is_leaf_node()) {
                        nested = !covered[static_cast<size_t>(j - begin)] && j > last_leaf;
                        last_leaf = j;
                    }
                    nested = nested && contains(header.quantized_aabb_min, header.quantized_aabb_max,
                                                nodes[j].quantized_aabb_min, nodes[j].quantized_aabb_max);
                    covered[static_cast<size_t>(j - begin)] = 1;
                }
            }
            for (int j = begin; nested && j < end; ++j) {
                nested = !nodes[j].is_leaf_node() || covered[static_cast<size_t>(j - begin)] != 0;
            }
        }
        if (!nested) {
            continue;
        }

        const Vec3 q = quantization[static_cast<size_t>(mesh.mesh_index)].bvh_quantization;
        ArenaMeshLeafGridInfo& info = out_infos[m];
        const float scale[3] = {q.x, q.y, q.z};
        int max_q[3] = {0, 0, 0};
        for (int i = begin; i < end; ++i) {
            for (int a = 0; a < 3 && nodes[i].is_leaf_node(); ++a) {
                max_q[a] = std::max(max_q[a], static_cast<int>(nodes[i].quantized_aabb_max[a]));
            }
        }
        for (int a = 0; a < 3; ++a) {
            info.cell_width[a] = std::max(1, static_cast<int>(std::ceil(kCellBt * scale[a])));
            info.cells[a] = max_q[a] / info.cell_width[a] + 1;
#if !defined(ROCKETGPU_USE_CUDA)
            info.cell_magic[a] = (uint64_t{1} << 32) / static_cast<uint32_t>(info.cell_width[a]) + 1u;
#endif
        }
        const size_t cell_count = static_cast<size_t>(info.cells[0]) * info.cells[1] * info.cells[2];
        std::vector<std::vector<ArenaMeshLeafEntry>> cells(cell_count);
        for (int i = begin; i < end; ++i) {
            const ArenaMeshBvhNode& node = nodes[i];
            if (!node.is_leaf_node()) {
                continue;
            }
            ArenaMeshLeafEntry entry;
            int lo[3];
            int hi[3];
            for (int a = 0; a < 3; ++a) {
                entry.quantized_aabb_min[a] = node.quantized_aabb_min[a];
                entry.quantized_aabb_max[a] = node.quantized_aabb_max[a];
                lo[a] = std::max(0, node.quantized_aabb_min[a] / info.cell_width[a] - 1);
                hi[a] = std::min(info.cells[a] - 1, node.quantized_aabb_max[a] / info.cell_width[a]);
            }
            entry.triangle_index = node.get_triangle_index();
            for (int z = lo[2]; z <= hi[2]; ++z) {
                for (int y = lo[1]; y <= hi[1]; ++y) {
                    for (int x = lo[0]; x <= hi[0]; ++x) {
                        cells[(static_cast<size_t>(z) * info.cells[1] + y) * info.cells[0] + x].push_back(entry);
                    }
                }
            }
        }
        info.cell_offset = static_cast<int32_t>(out_cell_start.size()) - 1;
        for (const std::vector<ArenaMeshLeafEntry>& cell : cells) {
            out_entries.insert(out_entries.end(), cell.begin(), cell.end());
            out_cell_start.push_back(static_cast<int32_t>(out_entries.size()));
        }
    }
}
}
}
