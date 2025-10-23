#include "Ray.cuh"


namespace LR {
    __device__ Ray::Ray(const Vec3& origin, const Vec3& direction) : m_origin(origin), m_dir(direction) {}

    __device__ Vec3 Ray::getOrigin() const {
        return m_origin;
    }

    __device__ Vec3 Ray::getDirection() const {
        return m_dir;
    }

    __device__ Vec3 Ray::pointAt(float t) {
        return m_origin + t * m_dir;
    }
}