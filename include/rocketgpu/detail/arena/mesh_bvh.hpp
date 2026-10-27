#pragma once

#include <cstdint>
#include <cstring>

#include "rocketgpu/detail/constants.hpp"

namespace rocketgpu {
namespace detail {
struct ArenaMeshTriangle {
    Vec3 a = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 b = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 c = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 normal = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 min = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 max = Vec3(0.0f, 0.0f, 0.0f);
    float edge_v0v1_angle = 2.0f * Pi();
    float edge_v1v2_angle = 2.0f * Pi();
    float edge_v2v0_angle = 2.0f * Pi();
    uint8_t edge_flags = 0;
    uint8_t mesh_index = 0;
    uint16_t quantized_aabb_min[3] = {0, 0, 0};
    uint16_t quantized_aabb_max[3] = {0, 0, 0};

    int8_t cal_origin_sign = 0;
};

constexpr float kGoalMeshOriginYBt = 102.39998626708984f;

struct ArenaMeshBvhQuantization {
    Vec3 bvh_aabb_min = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 bvh_aabb_max = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 bvh_quantization = Vec3(1.0f, 1.0f, 1.0f);
};

// 16-byte aligned so the device walk fetches a node with one 128-bit load.
struct alignas(16) ArenaMeshBvhNode {
    uint16_t quantized_aabb_min[3] = {0, 0, 0};
    uint16_t quantized_aabb_max[3] = {0, 0, 0};
    int32_t escape_index_or_triangle_index = 0;

    ROCKETGPU_HD bool is_leaf_node() const {
        return escape_index_or_triangle_index >= 0;
    }

    ROCKETGPU_HD int32_t get_escape_index() const {
        return -escape_index_or_triangle_index;
    }

    ROCKETGPU_HD int32_t get_triangle_index() const {
        return escape_index_or_triangle_index;
    }
};
static_assert(sizeof(ArenaMeshBvhNode) == 16, "ArenaMeshBvhNode must stay one 128-bit load");

struct ArenaMeshBvhSubtreeInfo {
    uint16_t quantized_aabb_min[3] = {0, 0, 0};
    uint16_t quantized_aabb_max[3] = {0, 0, 0};
    int32_t root_node_index = 0;
    int32_t subtree_size = 0;
};

struct ArenaBroadphaseGrid {
    Vec3 min_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 max_bt = Vec3(0.0f, 0.0f, 0.0f);
    float cell_size_bt = 0.0f;
    int32_t cells_x = 0;
    int32_t cells_y = 0;
    int32_t cells_z = 0;
    const uint16_t* cell_mesh_mask = nullptr;
    int32_t cell_count = 0;

    ROCKETGPU_HD bool valid() const {
        return cell_mesh_mask != nullptr && cell_count > 0 && cell_size_bt > 0.0f &&
               cells_x > 0 && cells_y > 0 && cells_z > 0;
    }

    ROCKETGPU_HD void cell_indices(const Vec3& pos_bt, int& i, int& j, int& k) const {
        i = static_cast<int>((pos_bt.x - min_bt.x) / cell_size_bt);
        j = static_cast<int>((pos_bt.y - min_bt.y) / cell_size_bt);
        k = static_cast<int>((pos_bt.z - min_bt.z) / cell_size_bt);
        if (i < 0) i = 0;
        if (j < 0) j = 0;
        if (k < 0) k = 0;
        if (i > cells_x - 1) i = cells_x - 1;
        if (j > cells_y - 1) j = cells_y - 1;
        if (k > cells_z - 1) k = cells_z - 1;
    }

    ROCKETGPU_HD int cell_index(const Vec3& pos_bt) const {
        int i = 0;
        int j = 0;
        int k = 0;
        cell_indices(pos_bt, i, j, k);
        return i * cells_y * cells_z + j * cells_z + k;
    }

    ROCKETGPU_HD bool mesh_registered_in_cell(int cell_index_value, int mesh_index) const {
        if (!valid() || cell_index_value < 0 || cell_index_value >= cell_count ||
            mesh_index < 0 || mesh_index >= 16) {
            return false;
        }
        return (cell_mesh_mask[cell_index_value] & static_cast<uint16_t>(1u << mesh_index)) != 0;
    }
};

struct ArenaMeshBvhMeshInfo {
    Vec3 aabb_min = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 aabb_max = Vec3(0.0f, 0.0f, 0.0f);
    int32_t mesh_index = -1;
    int32_t triangle_start = 0;
    int32_t triangle_count = 0;
    int32_t node_offset = 0;
    int32_t node_count = 0;
    int32_t subtree_offset = 0;
    int32_t subtree_count = 0;
    // The same subtree_count headers sorted by root_node_index (cal_bvh_dfs_order). Headers partition the
    // leaves and a parent's quantized AABB contains its children's, so the cache-friendly walk over these
    // visits exactly the leaves of the whole-tree stackless DFS, in the same (node index) order. -1: none.
    int32_t dfs_subtree_offset = -1;

    float cal_origin_y = 0.0f;
};

enum ArenaTriangleEdgeFlags : uint8_t {
    ROCKETGPU_TRI_INFO_V0V1_CONVEX = 1 << 0,
    ROCKETGPU_TRI_INFO_V1V2_CONVEX = 1 << 1,
    ROCKETGPU_TRI_INFO_V2V0_CONVEX = 1 << 2,
    ROCKETGPU_TRI_INFO_V0V1_SWAP_NORMALB = 1 << 3,
    ROCKETGPU_TRI_INFO_V1V2_SWAP_NORMALB = 1 << 4,
    ROCKETGPU_TRI_INFO_V2V0_SWAP_NORMALB = 1 << 5,
};

ROCKETGPU_HD Vec3 MinComponents(const Vec3& a, const Vec3& b) {
    return Vec3(Min(a.x, b.x), Min(a.y, b.y), Min(a.z, b.z));
}

ROCKETGPU_HD Vec3 MaxComponents(const Vec3& a, const Vec3& b) {
    return Vec3(Max(a.x, b.x), Max(a.y, b.y), Max(a.z, b.z));
}

ROCKETGPU_HD Vec3 MulComponents(const Vec3& a, const Vec3& b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

ROCKETGPU_HD Vec3 DivComponents(const Vec3& a, const Vec3& b) {
    return Vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}

ROCKETGPU_HD void QuantizeArenaMeshBvhPoint(
    const ArenaMeshBvhQuantization& quantization,
    const Vec3& point,
    int is_max,
    uint16_t out[3]);

ROCKETGPU_HD Vec3 UnquantizeArenaMeshBvhPoint(
    const ArenaMeshBvhQuantization& quantization,
    const uint16_t in[3]);

ROCKETGPU_HD bool TestQuantizedAabbAgainstQuantizedAabb(
    const uint16_t* aabb_min_1,
    const uint16_t* aabb_max_1,
    const uint16_t* aabb_min_2,
    const uint16_t* aabb_max_2);

ROCKETGPU_HD void SetArenaMeshBvhQuantization(
    ArenaMeshBvhQuantization& out_quantization,
    const Vec3& bvh_aabb_min,
    const Vec3& bvh_aabb_max,
    float quantization_margin = 1.0f,
    bool legacy_282 = false) {
    const Vec3 clamp_value(quantization_margin, quantization_margin, quantization_margin);
    out_quantization.bvh_aabb_min = bvh_aabb_min - clamp_value;
    out_quantization.bvh_aabb_max = bvh_aabb_max + clamp_value;

    Vec3 aabb_size = out_quantization.bvh_aabb_max - out_quantization.bvh_aabb_min;
    out_quantization.bvh_quantization = DivComponents(Vec3(65533.0f, 65533.0f, 65533.0f), aabb_size);

    if (legacy_282) {
        uint16_t quantized_min[3];
        const Vec3 current_min = out_quantization.bvh_aabb_min;
        QuantizeArenaMeshBvhPoint(out_quantization, current_min, 0, quantized_min);
        const Vec3 unquantized_min(
            static_cast<float>(quantized_min[0]) / out_quantization.bvh_quantization.x,
            static_cast<float>(quantized_min[1]) / out_quantization.bvh_quantization.y,
            static_cast<float>(quantized_min[2]) / out_quantization.bvh_quantization.z);
        out_quantization.bvh_aabb_min = MinComponents(current_min, current_min + unquantized_min - clamp_value);
        const Vec3 new_min = out_quantization.bvh_aabb_min;
        const Vec3 current_max = out_quantization.bvh_aabb_max;
        const float qx = (current_max.x - new_min.x) * out_quantization.bvh_quantization.x;
        const float qy = (current_max.y - new_min.y) * out_quantization.bvh_quantization.y;
        const float qz = (current_max.z - new_min.z) * out_quantization.bvh_quantization.z;
        const uint16_t qmax[3] = {
            static_cast<uint16_t>(static_cast<uint16_t>(static_cast<int>(qx + 1.0f)) | 1),
            static_cast<uint16_t>(static_cast<uint16_t>(static_cast<int>(qy + 1.0f)) | 1),
            static_cast<uint16_t>(static_cast<uint16_t>(static_cast<int>(qz + 1.0f)) | 1)};
        const Vec3 unquantized_max(
            static_cast<float>(qmax[0]) / out_quantization.bvh_quantization.x,
            static_cast<float>(qmax[1]) / out_quantization.bvh_quantization.y,
            static_cast<float>(qmax[2]) / out_quantization.bvh_quantization.z);
        out_quantization.bvh_aabb_max = MaxComponents(current_max, new_min + unquantized_max + clamp_value);
        aabb_size = out_quantization.bvh_aabb_max - out_quantization.bvh_aabb_min;
        out_quantization.bvh_quantization = DivComponents(Vec3(65533.0f, 65533.0f, 65533.0f), aabb_size);
        return;
    }

    uint16_t quantized_bounds[3];
    {
        const Vec3 current_min = out_quantization.bvh_aabb_min;
        QuantizeArenaMeshBvhPoint(out_quantization, out_quantization.bvh_aabb_min, 0, quantized_bounds);
        const Vec3 unquantized_min(
            static_cast<float>(quantized_bounds[0]) / out_quantization.bvh_quantization.x,
            static_cast<float>(quantized_bounds[1]) / out_quantization.bvh_quantization.y,
            static_cast<float>(quantized_bounds[2]) / out_quantization.bvh_quantization.z);
        out_quantization.bvh_aabb_min = MinComponents(current_min, current_min + unquantized_min - clamp_value);
    }

    aabb_size = out_quantization.bvh_aabb_max - out_quantization.bvh_aabb_min;
    out_quantization.bvh_quantization = DivComponents(Vec3(65533.0f, 65533.0f, 65533.0f), aabb_size);

    {
        const Vec3 current_min = out_quantization.bvh_aabb_min;
        const Vec3 current_max = out_quantization.bvh_aabb_max;
        QuantizeArenaMeshBvhPoint(out_quantization, out_quantization.bvh_aabb_max, 1, quantized_bounds);
        const Vec3 unquantized_max(
            static_cast<float>(quantized_bounds[0]) / out_quantization.bvh_quantization.x,
            static_cast<float>(quantized_bounds[1]) / out_quantization.bvh_quantization.y,
            static_cast<float>(quantized_bounds[2]) / out_quantization.bvh_quantization.z);
        out_quantization.bvh_aabb_max = MaxComponents(current_max, current_min + unquantized_max + clamp_value);
    }

    aabb_size = out_quantization.bvh_aabb_max - out_quantization.bvh_aabb_min;
    out_quantization.bvh_quantization = DivComponents(Vec3(65533.0f, 65533.0f, 65533.0f), aabb_size);
}

ROCKETGPU_HD void QuantizeArenaMeshBvhPoint(
    const ArenaMeshBvhQuantization& quantization,
    const Vec3& point,
    int is_max,
    uint16_t out[3]) {
    Vec3 clamped_point = point;
    clamped_point = MaxComponents(clamped_point, quantization.bvh_aabb_min);
    clamped_point = MinComponents(clamped_point, quantization.bvh_aabb_max);
    const Vec3 v = MulComponents(clamped_point - quantization.bvh_aabb_min, quantization.bvh_quantization);
    if (is_max) {
        out[0] = static_cast<uint16_t>((static_cast<uint16_t>(v.x + 1.0f)) | 1u);
        out[1] = static_cast<uint16_t>((static_cast<uint16_t>(v.y + 1.0f)) | 1u);
        out[2] = static_cast<uint16_t>((static_cast<uint16_t>(v.z + 1.0f)) | 1u);
    } else {
        out[0] = static_cast<uint16_t>(static_cast<uint16_t>(v.x) & 0xfffeu);
        out[1] = static_cast<uint16_t>(static_cast<uint16_t>(v.y) & 0xfffeu);
        out[2] = static_cast<uint16_t>(static_cast<uint16_t>(v.z) & 0xfffeu);
    }
}

ROCKETGPU_HD Vec3 UnquantizeArenaMeshBvhPoint(
    const ArenaMeshBvhQuantization& quantization,
    const uint16_t in[3]) {
    return Vec3(
        quantization.bvh_aabb_min.x + static_cast<float>(in[0]) / quantization.bvh_quantization.x,
        quantization.bvh_aabb_min.y + static_cast<float>(in[1]) / quantization.bvh_quantization.y,
        quantization.bvh_aabb_min.z + static_cast<float>(in[2]) / quantization.bvh_quantization.z);
}

ROCKETGPU_HD bool TestQuantizedAabbAgainstQuantizedAabb(
    const uint16_t* aabb_min_1,
    const uint16_t* aabb_max_1,
    const uint16_t* aabb_min_2,
    const uint16_t* aabb_max_2) {
    return !(aabb_min_1[0] > aabb_max_2[0] || aabb_max_1[0] < aabb_min_2[0] ||
             aabb_min_1[1] > aabb_max_2[1] || aabb_max_1[1] < aabb_min_2[1] ||
             aabb_min_1[2] > aabb_max_2[2] || aabb_max_1[2] < aabb_min_2[2]);
}

#if defined(__CUDA_ARCH__)
__device__ inline ArenaMeshBvhNode LoadArenaMeshBvhNode(const ArenaMeshBvhNode* p) {
    ArenaMeshBvhNode node;
    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(p));
    memcpy(&node, &raw, sizeof(node));
    return node;
}
#endif

template <typename Callback>
ROCKETGPU_HD void WalkArenaMeshBvhStacklessQuantizedTree(
    const ArenaMeshBvhNode* nodes,
    const uint16_t* quantized_query_aabb_min,
    const uint16_t* quantized_query_aabb_max,
    int start_node_index,
    int end_node_index,
    Callback&& callback) {
    int cur_index = start_node_index;
    const ArenaMeshBvhNode* root_node = &nodes[start_node_index];
#if defined(__CUDA_ARCH__)
    // The next node in memory is fetched while the current one is tested, so the common descend step
    // does not wait on a dependent load.
    ArenaMeshBvhNode node = cur_index < end_node_index ? LoadArenaMeshBvhNode(root_node) : ArenaMeshBvhNode{};
    while (cur_index < end_node_index) {
        const ArenaMeshBvhNode next = cur_index + 1 < end_node_index ? LoadArenaMeshBvhNode(root_node + 1) : node;
#else
    while (cur_index < end_node_index) {
        const ArenaMeshBvhNode& node = *root_node;
#endif
        const bool aabb_overlap = TestQuantizedAabbAgainstQuantizedAabb(
            quantized_query_aabb_min,
            quantized_query_aabb_max,
            node.quantized_aabb_min,
            node.quantized_aabb_max);
        const bool is_leaf_node = node.is_leaf_node();
        if (is_leaf_node && aabb_overlap) {
            callback(node.get_triangle_index());
        }

        if (aabb_overlap || is_leaf_node) {
            ++root_node;
            ++cur_index;
#if defined(__CUDA_ARCH__)
            node = next;
#endif
        } else {
            const int escape_index = node.get_escape_index();
            root_node += escape_index;
            cur_index += escape_index;
#if defined(__CUDA_ARCH__)
            if (cur_index < end_node_index) {
                node = LoadArenaMeshBvhNode(root_node);
            }
#endif
        }
    }
}

// cal_bvh_dfs_order (stock btQuantizedBvh TRAVERSAL_STACKLESS, whole-tree DFS): switch a mesh's headers to its
// node-index-sorted copy, whose cache-friendly walk visits the same leaves in the same order; without that copy,
// count 0 makes the walk take the whole tree.
ROCKETGPU_HD void SelectArenaMeshBvhDfsOrderSubtrees(
    const ArenaMeshBvhMeshInfo& mesh_info,
    const ArenaMeshBvhSubtreeInfo* all_subtrees,
    int all_subtree_count,
    const ArenaMeshBvhSubtreeInfo*& mesh_subtrees,
    int& mesh_subtree_count) {
    if (mesh_subtrees != nullptr && mesh_subtree_count > 0 && mesh_info.dfs_subtree_offset >= 0 &&
        mesh_info.dfs_subtree_offset + mesh_subtree_count <= all_subtree_count) {
        mesh_subtrees = all_subtrees + mesh_info.dfs_subtree_offset;
        return;
    }
    mesh_subtree_count = 0;
}

template <typename Callback>
ROCKETGPU_HD void WalkArenaMeshBvhStacklessQuantizedTreeCacheFriendly(
    const ArenaMeshBvhNode* nodes,
    const ArenaMeshBvhSubtreeInfo* subtrees,
    int subtree_count,
    const uint16_t* quantized_query_aabb_min,
    const uint16_t* quantized_query_aabb_max,
    int root_node_index,
    int node_count,
    Callback&& callback) {
    if (subtrees != nullptr && subtree_count > 0) {
        for (int subtree_idx = 0; subtree_idx < subtree_count; ++subtree_idx) {
            const ArenaMeshBvhSubtreeInfo& subtree = subtrees[subtree_idx];
            const bool overlap = TestQuantizedAabbAgainstQuantizedAabb(
                quantized_query_aabb_min,
                quantized_query_aabb_max,
                subtree.quantized_aabb_min,
                subtree.quantized_aabb_max);
            if (!overlap) {
                continue;
            }
            WalkArenaMeshBvhStacklessQuantizedTree(
                nodes,
                quantized_query_aabb_min,
                quantized_query_aabb_max,
                subtree.root_node_index,
                subtree.root_node_index + subtree.subtree_size,
                callback);
        }
        return;
    }

    WalkArenaMeshBvhStacklessQuantizedTree(
        nodes,
        quantized_query_aabb_min,
        quantized_query_aabb_max,
        root_node_index,
        root_node_index + node_count,
        callback);
}
}
}
