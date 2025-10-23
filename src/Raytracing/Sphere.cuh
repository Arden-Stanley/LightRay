#ifndef SPHERE_H
#define SPHERE_H

#include "Vector.cuh"
#include "Ray.cuh"

namespace LR {
    class Sphere {
        public:
            __device__ Sphere(float r, const Vec3& pos);
            __device__ bool checkHit(const Ray &ray) const;
        private:
            float m_radius;
            Vec3 m_position;
    };
}

#endif