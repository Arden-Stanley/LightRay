#include "Ray.cuh"

namespace LR {
        __device__ Ray::Ray(const Vec3& origin, const Vec3& direction) : m_origin(origin), m_dir(direction) {}

        __device__ Vec3 Ray::Origin() const {
            return m_origin;
        }

        __device__ Vec3 Ray::Direction() const {
            return m_dir;
        }

        __device__ Vec3 Ray::PointAt(float t) const {
            return m_origin + t * m_dir;
        }
}