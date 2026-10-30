#include "rocketgpu/detail/arena/mesh_loader.hpp"
#include "rocketgpu/detail/arena/mesh_path.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
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
    return atan2f(Dot(swing_axis, ref_axis0), Dot(swing_axis, ref_axis1));
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
            for (int axis = 0; axis < 3; ++axis) {
                if (internal_node.quantized_aabb_min[axis] > leaf.quantized_aabb_min[axis]) {
                    internal_node.quantized_aabb_min[axis] = leaf.quantized_aabb_min[axis];
                }
                if (internal_node.quantized_aabb_max[axis] < leaf.quantized_aabb_max[axis]) {
                    internal_node.quantized_aabb_max[axis] = leaf.quantized_aabb_max[axis];
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

        if (out_quantization) {
            SetArenaMeshBvhQuantization((*out_quantization)[static_cast<size_t>(mesh_idx)], mesh_min, mesh_max, 1.0f, out_local_triangles != nullptr);
            const ArenaMeshBvhQuantization& quantization = (*out_quantization)[static_cast<size_t>(mesh_idx)];
            for (size_t tri_pos = mesh_base; tri_pos < triangles.size(); ++tri_pos) {
                ArenaMeshTriangle& tri = triangles[tri_pos];
                QuantizeArenaMeshBvhPoint(quantization, tri.min, 0, tri.quantized_aabb_min);
                QuantizeArenaMeshBvhPoint(quantization, tri.max, 1, tri.quantized_aabb_max);
            }

            if (out_mesh_infos && out_nodes && out_subtrees) {
                std::vector<ArenaMeshBvhNode> mesh_nodes;
                std::vector<ArenaMeshBvhSubtreeInfo> mesh_subtrees;
                BuildArenaMeshBulletStyleBvh(
                    triangles,
                    static_cast<int32_t>(mesh_base),
                    num_tris,
                    quantization,
                    mesh_nodes,
                    mesh_subtrees);

                ArenaMeshBvhMeshInfo mesh_info;
                mesh_info.aabb_min = mesh_min;
                mesh_info.aabb_max = mesh_max;
                mesh_info.mesh_index = mesh_idx;
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
                // cal_bvh_dfs_order headers (ArenaMeshBvhMeshInfo::dfs_subtree_offset): node-index order.
                std::stable_sort(mesh_subtrees.begin(), mesh_subtrees.end(),
                                 [](const ArenaMeshBvhSubtreeInfo& a, const ArenaMeshBvhSubtreeInfo& b) {
                                     return a.root_node_index < b.root_node_index;
                                 });
                out_subtrees->insert(out_subtrees->end(), mesh_subtrees.begin(), mesh_subtrees.end());
            }
        }

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

        if (out_local_triangles) {
            // The game's goal meshes are in their own frame, placed at the goal: a mesh centered past the goal line
            // is moved into it
            const float center_y = 0.5f * (mesh_min.y + mesh_max.y);
            const int8_t origin_sign = center_y > kGoalMeshOriginYBt ? 1 : (center_y < -kGoalMeshOriginYBt ? -1 : 0);
            std::vector<MeshVertex> frame_vertices = vertices;
            for (MeshVertex& v : frame_vertices) v.y -= static_cast<float>(origin_sign) * kGoalMeshOriginYBt;
            local_goal_meshes += origin_sign != 0;
            if (out_mesh_infos && !out_mesh_infos->empty() && out_mesh_infos->back().mesh_index == mesh_idx) {
                out_mesh_infos->back().cal_origin_y = static_cast<float>(origin_sign) * kGoalMeshOriginYBt;
            }

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

void SetArenaMeshFreeBox(std::vector<ArenaMeshBvhMeshInfo>& mesh_infos) {
    // A mesh's gap on an axis: how far its AABB stays from 0 on that axis (0 when it straddles 0).
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
        // Meshes whose x gap reaches x lie beyond the box in x; every other one must lie beyond it in y.
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

void BuildArenaMeshLeafGrid(
    const std::vector<ArenaMeshBvhQuantization>& quantization,
    const std::vector<ArenaMeshBvhMeshInfo>& mesh_infos,
    const std::vector<ArenaMeshBvhNode>& nodes,
    const std::vector<ArenaMeshBvhSubtreeInfo>& subtrees,
    std::vector<ArenaMeshLeafGridInfo>& out_infos,
    std::vector<int32_t>& out_cell_start,
    std::vector<ArenaMeshLeafEntry>& out_entries) {
    // A car's mesh query AABB is at most ~3 bt wide; wider queries walk the BVH.
    constexpr float kCellBt = 4.0f;
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
        // The walk visits a leaf when it and every enclosing node (and subtree header) overlap the query.
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
        // The walk takes the node-index-sorted headers when it has them (SelectArenaMeshBvhDfsOrderSubtrees),
        // else the whole tree; the headers must then cover every leaf.
        const bool headers = mesh.subtree_count > 0 && mesh.subtree_offset >= 0 && mesh.dfs_subtree_offset >= 0 &&
                             mesh.dfs_subtree_offset + mesh.subtree_count <= static_cast<int>(subtrees.size());
        if (nested && headers) {
            std::vector<uint8_t> covered(static_cast<size_t>(mesh.node_count), 0);
            for (int h = 0; nested && h < mesh.subtree_count; ++h) {
                const ArenaMeshBvhSubtreeInfo& header = subtrees[static_cast<size_t>(mesh.dfs_subtree_offset + h)];
                const int stop = std::min(end, header.root_node_index + header.subtree_size);
                for (int j = std::max(begin, header.root_node_index); nested && j < stop; ++j) {
                    nested = contains(header.quantized_aabb_min, header.quantized_aabb_max,
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
