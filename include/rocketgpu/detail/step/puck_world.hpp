

struct PuckPlaneContactBt {
    uint8_t has_collision = 0;
    Vec3 normal_on_b = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 point_on_b_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 point_on_a_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 local_point_a_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 local_point_b_bt = Vec3(0.0f, 0.0f, 0.0f);
    float distance_bt = 0.0f;
    int support_index = -1;
};
