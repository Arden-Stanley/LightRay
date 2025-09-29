#pragma once
#include <vector>

struct Sphere {
    float x, y, z;
    float radius;
};

class Scene {

public:
    float fieldOfView = 45.0f;
    int rayDepth = 2;
    std::vector<Sphere> spheres;
    Scene() = default;

    void Reset() {
        spheres.clear();
        fieldOfView = 45.0f;
        rayDepth = 2;
    }

    void AddSphere(float x = 0.0f, float y = 0.0f, float z = -5.0f, float radius = 1.0f) {
        spheres.push_back({ x, y, z, radius });
    }
};