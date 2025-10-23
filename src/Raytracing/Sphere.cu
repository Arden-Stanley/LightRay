#include "Sphere.cuh"

namespace LR {
    __device__ Sphere::Sphere(float r, const Vec3 &pos) : m_radius(r), m_position(pos) {}
    
    __device__ bool Sphere::checkHit(const Ray &r) const {
        Vec3 oc = m_position - r.getOrigin();
        float a = dot(r.getDirection(), r.getDirection());
        float b = -2.0f * dot(r.getDirection(), oc);
        float c = dot(oc, oc) - m_radius * m_radius;
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {
            return false;
        }
        else {
            return true;
        }
    }
}