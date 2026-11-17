#include <cstdio>
#include <cstring>

#include <rocketgpu/rocketgpu.h>

#include "shim.h"

static RgCarConfig Convert(const rocketgpu::CarConfig& c) {
    RgCarConfig o{};
    const rocketgpu::Vec3* v[5] = {&c.hitbox_size, &c.hitbox_published, &c.hitbox_offset, &c.front_wheels_offset,
                                   &c.back_wheels_offset};
    float* f[5] = {o.hitbox_size, o.hitbox_published, o.hitbox_offset, o.front_wheels_offset, o.back_wheels_offset};
    for (int i = 0; i < 5; ++i) {
        f[i][0] = v[i]->x;
        f[i][1] = v[i]->y;
        f[i][2] = v[i]->z;
    }
    o.front_wheel_radius = c.front_wheel_radius;
    o.back_wheel_radius = c.back_wheel_radius;
    o.front_suspension_rest = c.front_suspension_rest;
    o.back_suspension_rest = c.back_suspension_rest;
    o.dodge_deadzone = c.dodge_deadzone;
    std::memcpy(o.steer_angle_curve_deg, c.steer_angle_curve_deg, sizeof(o.steer_angle_curve_deg));
    std::memcpy(o.powerslide_steer_angle_curve_deg, c.powerslide_steer_angle_curve_deg,
                sizeof(o.powerslide_steer_angle_curve_deg));
    o.spawn_z = c.spawn_z;
    o.body = c.body;
    o.three_wheels = c.three_wheels;
    return o;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const rocketgpu::CarConfig presets[8] = {
        rocketgpu::OctaneConfig(), rocketgpu::DominusConfig(), rocketgpu::PlankConfig(), rocketgpu::BreakoutConfig(),
        rocketgpu::HybridConfig(), rocketgpu::MercConfig(),    rocketgpu::PsyclopsConfig(), rocketgpu::BatmobileConfig()};
    std::FILE* f = std::fopen(argv[1], "w");
    if (!f) return 1;
    std::fprintf(f, "pub(crate) const PRESET_WORDS: [[u32; 31]; 8] = [\n");
    for (const rocketgpu::CarConfig& preset : presets) {
        const RgCarConfig c = Convert(preset);
        uint32_t words[31];
        std::memcpy(words, &c, sizeof(words));
        std::fprintf(f, "    [");
        for (uint32_t w : words) std::fprintf(f, "0x%08x, ", w);
        std::fprintf(f, "],\n");
    }
    std::fprintf(f, "];\n");
    return std::fclose(f) == 0 ? 0 : 1;
}
