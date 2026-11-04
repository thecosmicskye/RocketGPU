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

struct ArenaMeshLeafGridInfo {
    int32_t cell_offset = -1;
    int32_t cells[3] = {0, 0, 0};
    int32_t cell_width[3] = {1, 1, 1};
};

struct alignas(16) ArenaMeshLeafEntry {
    uint16_t quantized_aabb_min[3] = {0, 0, 0};
    uint16_t quantized_aabb_max[3] = {0, 0, 0};
    int32_t triangle_index = 0;
};
static_assert(sizeof(ArenaMeshLeafEntry) == 16, "ArenaMeshLeafEntry must stay one 128-bit load");

struct ArenaMeshLeafGrid {
    const ArenaMeshLeafGridInfo* infos = nullptr;
    const int32_t* cell_start = nullptr;
    const ArenaMeshLeafEntry* entries = nullptr;
};

struct BulletSweepHistory;

struct ArenaBroadphaseGrid {
    Vec3 min_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 max_bt = Vec3(0.0f, 0.0f, 0.0f);
    float cell_size_bt = 0.0f;
    int32_t cells_x = 0;
    int32_t cells_y = 0;
    int32_t cells_z = 0;
    const uint16_t* cell_mesh_mask = nullptr;
    int32_t cell_count = 0;
    ArenaMeshLeafGrid leaf_grid;
    const BulletSweepHistory* sweep_statics = nullptr;

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
    int32_t dfs_subtree_offset = -1;

    float cal_origin_y = 0.0f;
    float free_half_x = 0.0f;
    float free_half_y = 0.0f;
    int32_t ray_order = -1;
    int32_t ray_neg_x_plane_pos = 0;
};

constexpr float kArenaSweepHalfExtentBt = 10485.740234375f;
ROCKETGPU_HD uint16_t ArenaSweepQuantize(float v, int is_max) {
    const float sap_quantize = 65535.0f / (kArenaSweepHalfExtentBt + kArenaSweepHalfExtentBt);
    float scaled = (v + kArenaSweepHalfExtentBt) * sap_quantize;
    scaled = scaled >= 0.0f ? (scaled < 65535.0f ? scaled : 65535.0f) : 0.0f;
    return static_cast<uint16_t>((static_cast<int>(scaled) & 0xfffe) | is_max);
}

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

ROCKETGPU_HD void QuantizeArenaMeshBvhQueryBox(
    const ArenaMeshBvhQuantization& quantization,
    const Vec3& center,
    const Vec3& extent,
    float origin_y,
    uint16_t out_min[3],
    uint16_t out_max[3]) {
    const Vec3 c(center.x, center.y - origin_y, center.z);
    QuantizeArenaMeshBvhPoint(quantization, c - extent, 0, out_min);
    QuantizeArenaMeshBvhPoint(quantization, c + extent, 1, out_max);
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

#ifndef ROCKETGPU_WALK_WINDOW
#define ROCKETGPU_WALK_WINDOW 4
#endif

struct ArenaMeshBvhAnyLeaf {
    ROCKETGPU_HD bool operator()(const uint16_t*, const uint16_t*) const { return true; }
};

struct ArenaMeshBvhRayLeaf {
    const ArenaMeshBvhQuantization* quantization;
    Vec3 source;
    Vec3 to;
    mutable Vec3 inv_dir;
    mutable float lambda_max = -1.0f;

    ROCKETGPU_HD ArenaMeshBvhRayLeaf(const ArenaMeshBvhQuantization& q, const Vec3& from, const Vec3& ray_to)
        : quantization(&q), source(from), to(ray_to) {}

    ROCKETGPU_HD bool operator()(const uint16_t* qmin, const uint16_t* qmax) const {
        if (lambda_max < 0.0f) {
            const Vec3 d = to - source;
            const float r = 1.0f / sqrtf(d.y * d.y + d.x * d.x + d.z * d.z);
            const Vec3 n(d.x * r, d.y * r, d.z * r);
            lambda_max = n.x * d.x + n.y * d.y + n.z * d.z;
            inv_dir = Vec3(n.x == 0.0f ? 1e18f : 1.0f / n.x, n.y == 0.0f ? 1e18f : 1.0f / n.y,
                           n.z == 0.0f ? 1e18f : 1.0f / n.z);
        }
        const Vec3 lo = UnquantizeArenaMeshBvhPoint(*quantization, qmin);
        const Vec3 hi = UnquantizeArenaMeshBvhPoint(*quantization, qmax);
        const float tmin = ((inv_dir.x < 0.0f ? hi.x : lo.x) - source.x) * inv_dir.x;
        const float tmax = ((inv_dir.x < 0.0f ? lo.x : hi.x) - source.x) * inv_dir.x;
        const float tymin = ((inv_dir.y < 0.0f ? hi.y : lo.y) - source.y) * inv_dir.y;
        const float tymax = ((inv_dir.y < 0.0f ? lo.y : hi.y) - source.y) * inv_dir.y;
        if (!(tmin <= tymax) || !(tymin <= tmax)) {
            return false;
        }
        const float t0 = tymin <= tmin ? tmin : tymin;
        const float t1 = tmax <= tymax ? tmax : tymax;
        const float tzmin = ((inv_dir.z < 0.0f ? hi.z : lo.z) - source.z) * inv_dir.z;
        const float tzmax = ((inv_dir.z < 0.0f ? lo.z : hi.z) - source.z) * inv_dir.z;
        if (!(t0 <= tzmax) || !(tzmin <= t1)) {
            return false;
        }
        return (tzmin <= t0 ? t0 : tzmin) < lambda_max && 0.0f < (t1 <= tzmax ? t1 : tzmax);
    }
};

template <typename Callback, typename LeafTest = ArenaMeshBvhAnyLeaf>
ROCKETGPU_HD void WalkArenaMeshBvhStacklessQuantizedTree(
    const ArenaMeshBvhNode* nodes,
    const uint16_t* quantized_query_aabb_min,
    const uint16_t* quantized_query_aabb_max,
    int start_node_index,
    int end_node_index,
    Callback&& callback,
    const LeafTest& leaf_test = LeafTest{}) {
#if defined(__CUDA_ARCH__) && ROCKETGPU_WALK_WINDOW > 1
    {
        constexpr int W = ROCKETGPU_WALK_WINDOW;
        int cur = start_node_index;
        while (cur < end_node_index) {
            ArenaMeshBvhNode win[W];
#pragma unroll
            for (int k = 0; k < W; ++k) {
                win[k] = cur + k < end_node_index ? LoadArenaMeshBvhNode(nodes + cur + k) : ArenaMeshBvhNode{};
            }
            int pos = 0;
            unsigned hits = 0u;
            int32_t tri[W];
#pragma unroll
            for (int k = 0; k < W; ++k) {
                tri[k] = win[k].get_triangle_index();
                if (k == pos && cur + k < end_node_index) {
                    const bool overlap = TestQuantizedAabbAgainstQuantizedAabb(
                        quantized_query_aabb_min, quantized_query_aabb_max, win[k].quantized_aabb_min,
                        win[k].quantized_aabb_max);
                    const bool leaf = win[k].is_leaf_node();
                    if (leaf && overlap && leaf_test(win[k].quantized_aabb_min, win[k].quantized_aabb_max)) {
                        hits |= 1u << k;
                    }
                    pos = (overlap || leaf) ? k + 1 : k + win[k].get_escape_index();
                }
            }
            while (hits != 0u) {
                const int k = __ffs(static_cast<int>(hits)) - 1;
                hits &= hits - 1u;
                int32_t t = tri[0];
#pragma unroll
                for (int j = 1; j < W; ++j) {
                    t = k == j ? tri[j] : t;
                }
                callback(t);
            }
            cur += pos;
        }
        return;
    }
#endif
    int cur_index = start_node_index;
    const ArenaMeshBvhNode* root_node = &nodes[start_node_index];
#if defined(__CUDA_ARCH__)
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
        if (is_leaf_node && aabb_overlap && leaf_test(node.quantized_aabb_min, node.quantized_aabb_max)) {
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

ROCKETGPU_HD void SelectArenaMeshBvhDfsOrderSubtrees(
    const ArenaMeshBvhMeshInfo& mesh_info,
    const ArenaMeshBvhSubtreeInfo* all_subtrees,
    int all_subtree_count,
    const ArenaMeshBvhSubtreeInfo*& mesh_subtrees,
    int& mesh_subtree_count) {
#if !defined(__CUDA_ARCH__)
    (void)mesh_info;
    (void)all_subtrees;
    (void)all_subtree_count;
    (void)mesh_subtrees;
    mesh_subtree_count = 0;
#else
    if (mesh_subtrees != nullptr && mesh_subtree_count > 0 && mesh_info.dfs_subtree_offset >= 0 &&
        mesh_info.dfs_subtree_offset + mesh_subtree_count <= all_subtree_count) {
        mesh_subtrees = all_subtrees + mesh_info.dfs_subtree_offset;
        return;
    }
    mesh_subtree_count = 0;
#endif
}

ROCKETGPU_HD bool ValidateArenaMeshDfsOrderSubtrees(
    const ArenaMeshBvhMeshInfo& mesh_info,
    const ArenaMeshBvhNode* nodes,
    int node_count,
    const ArenaMeshBvhSubtreeInfo* subtrees,
    int subtree_count) {
    const int begin = mesh_info.node_offset;
    const int end = mesh_info.node_offset + mesh_info.node_count;
    if (nodes == nullptr || mesh_info.node_count <= 0 || begin < 0 || end > node_count) {
        return false;
    }
    if (mesh_info.subtree_count <= 0 || mesh_info.subtree_offset < 0 || mesh_info.dfs_subtree_offset < 0 ||
        subtrees == nullptr || mesh_info.dfs_subtree_offset + mesh_info.subtree_count > subtree_count) {
        return true;
    }
    int last_leaf_node = begin - 1;
    int header_leaves = 0;
    for (int h = 0; h < mesh_info.subtree_count; ++h) {
        const ArenaMeshBvhSubtreeInfo& header = subtrees[mesh_info.dfs_subtree_offset + h];
        const int h_begin = header.root_node_index;
        const int h_end = header.root_node_index + header.subtree_size;
        if (header.subtree_size <= 0 || h_begin < begin || h_end > end) {
            return false;
        }
        for (int j = h_begin; j < h_end; ++j) {
            if (nodes[j].is_leaf_node()) {
                if (j <= last_leaf_node) {
                    return false;
                }
                last_leaf_node = j;
                ++header_leaves;
            }
        }
    }
    int total_leaves = 0;
    for (int j = begin; j < end; ++j) {
        if (nodes[j].is_leaf_node()) {
            ++total_leaves;
        }
    }
    return header_leaves == total_leaves;
}

template <typename Callback, typename LeafTest = ArenaMeshBvhAnyLeaf>
ROCKETGPU_HD void WalkArenaMeshBvhStacklessQuantizedTreeCacheFriendly(
    const ArenaMeshBvhNode* nodes,
    const ArenaMeshBvhSubtreeInfo* subtrees,
    int subtree_count,
    const uint16_t* quantized_query_aabb_min,
    const uint16_t* quantized_query_aabb_max,
    int root_node_index,
    int node_count,
    Callback&& callback,
    const LeafTest& leaf_test = LeafTest{}) {
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
                callback,
                leaf_test);
        }
        return;
    }

    WalkArenaMeshBvhStacklessQuantizedTree(
        nodes,
        quantized_query_aabb_min,
        quantized_query_aabb_max,
        root_node_index,
        root_node_index + node_count,
        callback,
        leaf_test);
}

template <typename Callback, typename LeafTest = ArenaMeshBvhAnyLeaf>
ROCKETGPU_HD bool WalkArenaMeshLeafGrid(
    const ArenaMeshLeafGrid* grid,
    int mesh_info_idx,
    const uint16_t* quantized_query_aabb_min,
    const uint16_t* quantized_query_aabb_max,
    Callback&& callback,
    const LeafTest& leaf_test = LeafTest{}) {
    if (grid == nullptr || grid->infos == nullptr) {
        return false;
    }
    const ArenaMeshLeafGridInfo& info = grid->infos[mesh_info_idx];
    if (info.cell_offset < 0) {
        return false;
    }
    int cell[3];
    for (int a = 0; a < 3; ++a) {
        if (quantized_query_aabb_max[a] - quantized_query_aabb_min[a] > info.cell_width[a]) {
            return false;
        }
        cell[a] = quantized_query_aabb_min[a] / info.cell_width[a];
        cell[a] = cell[a] < info.cells[a] ? cell[a] : info.cells[a] - 1;
    }
    const int cell_idx = info.cell_offset + (cell[2] * info.cells[1] + cell[1]) * info.cells[0] + cell[0];
    const int end = grid->cell_start[cell_idx + 1];
    int e = grid->cell_start[cell_idx];
#if defined(__CUDA_ARCH__)
    constexpr int kLeafBatch = 8;
    for (; e + kLeafBatch <= end; e += kLeafBatch) {
        uint4 raw[kLeafBatch];
#pragma unroll
        for (int k = 0; k < kLeafBatch; ++k) {
            raw[k] = __ldg(reinterpret_cast<const uint4*>(&grid->entries[e + k]));
        }
        unsigned hits = 0u;
        int32_t triangles[kLeafBatch];
#pragma unroll
        for (int k = 0; k < kLeafBatch; ++k) {
            ArenaMeshLeafEntry entry;
            memcpy(&entry, &raw[k], sizeof(entry));
            triangles[k] = entry.triangle_index;
            if (TestQuantizedAabbAgainstQuantizedAabb(
                    quantized_query_aabb_min, quantized_query_aabb_max,
                    entry.quantized_aabb_min, entry.quantized_aabb_max) &&
                leaf_test(entry.quantized_aabb_min, entry.quantized_aabb_max)) {
                hits |= 1u << k;
            }
        }
        while (hits != 0u) {
            const int hit = __ffs(static_cast<int>(hits)) - 1;
            hits &= hits - 1u;
            int32_t triangle = triangles[0];
#pragma unroll
            for (int k = 1; k < kLeafBatch; ++k) {
                triangle = k == hit ? triangles[k] : triangle;
            }
            callback(triangle);
        }
    }
#endif
    for (; e < end; ++e) {
#if defined(__CUDA_ARCH__)
        ArenaMeshLeafEntry entry;
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(&grid->entries[e]));
        memcpy(&entry, &raw, sizeof(entry));
#else
        const ArenaMeshLeafEntry& entry = grid->entries[e];
#endif
        if (TestQuantizedAabbAgainstQuantizedAabb(
                quantized_query_aabb_min, quantized_query_aabb_max,
                entry.quantized_aabb_min, entry.quantized_aabb_max) &&
            leaf_test(entry.quantized_aabb_min, entry.quantized_aabb_max)) {
            callback(entry.triangle_index);
        }
    }
    return true;
}
}
}
