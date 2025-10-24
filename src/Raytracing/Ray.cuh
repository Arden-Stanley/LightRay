#ifndef RAY_CUH
#define RAY_CUH

#include "Vector.cuh"

namespace LR {    
    struct Payload {
        float t;
        Vec3 hit, normal;
    };

    class Ray {
        public:
            Payload payload;
            Ray() = default;
            __device__ Ray(const Vec3& origin, const Vec3& direction);
            __device__ Vec3 getOrigin() const;
            __device__ Vec3 getDirection() const;
            __device__ Vec3 pointAt(float t) const;
        private:
            Vec3 m_origin;
            Vec3 m_dir;

    };
}

#endif