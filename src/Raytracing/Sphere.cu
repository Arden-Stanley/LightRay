#include "Sphere.cuh"

namespace LR {
    __device__ Sphere::Sphere(float r, const Vec3 &pos) : m_radius(r), m_position(pos) {}
    
    __device__ bool Sphere::checkHit(Ray &r) const {
        Vec3 oc = m_position - r.getOrigin();
        float a = r.getDirection().lengthSqrd();
        float b = dot(r.getDirection(), oc);
        float c = oc.lengthSqrd() - m_radius * m_radius;

        float discriminant = b * b - a * c;
        if (discriminant < 0) {
            return false;
        }
        float sqrtd = sqrt(discriminant);

        float root = (b - sqrtd) / a;
        if (root <= 0.01f) {
            root = (b - sqrtd) / a;
            if (root <= 0.01f) {
                return false;
            }
        }

        setPayload(r, root);
        return true;
    }

    __device__ Vec3 Sphere::getCenter() const {
        return m_position;
    }

    __device__ float Sphere::getRadius() const {
        return m_radius;
    }

    __device__ void Sphere::setPayload(Ray &ray, float t) const {
        Vec3 hit = ray.pointAt(t);
        ray.payload = Payload(
            t, 
            hit,
            (hit - m_position) / m_radius
        );
    }
}