#ifndef RAY_CUH
#define RAY_CUH

#include "Vector.cuh"

namespace LR {
    class Ray {
        public:
            __device__ Ray() = default;
            __device__ Ray(const Vec3& origin, const Vec3& direction) : m_origin(origin), m_dir(direction) {}
            __device__ Vec3 getOrigin() const {return m_origin;}
            __device__ Vec3 getDirection() const {return m_dir;}
            __device__ Vec3 pointAt(float t) {return m_origin + t * m_dir;}
        private:
            Vec3 m_origin;
            Vec3 m_dir;
    };
}

#endif