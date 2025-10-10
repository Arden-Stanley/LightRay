#include "Scene.h"

namespace LR {
    Sphere::Sphere(const Vec3 &position, float radius)
    : m_position(position), m_radius(radius) {}

    Sphere::~Sphere() {}

    Scene::Scene() : _objects({}) {}

    Scene::~Scene() {}

    void Scene::AddObject(const Sphere &sphere) {
        _objects.push_back(sphere);
    }

    void Scene::Render() const {}
}