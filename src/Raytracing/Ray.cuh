#ifndef RAY_CUH
#define RAY_CUH

#include "Vector.cuh"

namespace LR {
    class Payload {
        public:
            Payload() = default;
            __device__ Payload(float t, Vec3 hit, Vec3 normal);
            __device__ void setT(float t);
            __device__ void setHit(const Vec3 &hit);
            __device__ void setNormal(const Vec3 &normal);
            __device__ float getT() const;
            __device__ Vec3 getHit() const;
            __device__ Vec3 getNormal() const;
        private:
            float m_t;
            Vec3 m_hit, m_normal;
    };

    class Ray {
        public:
            Payload payload;
            Ray() = default;
            __device__ Ray(const Vec3& origin, const Vec3& direction);
            __device__ Vec3 getOrigin() const;
            __device__ Vec3 getDirection() const;
            __device__ Vec3 pointAt(float t);
        private:
            Vec3 m_origin;
            Vec3 m_dir;
    };
}

#endif