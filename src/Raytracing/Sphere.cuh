#ifndef SPHERE_H
#define SPHERE_H

#include "Vector.cuh"
#include "Ray.cuh"
#include "Material.cuh"
#include <cuda_runtime.h>

namespace LR {
    namespace RT {
        class Sphere {
            public:
                __device__ Sphere(float r, const Vec3& pos, Material *material);
                __device__ ~Sphere();
                __device__ bool checkHit(Ray &ray) const;
                __device__ Vec3 getCenter() const;
                __device__ float getRadius() const;
                __device__ void setPayload(Ray &ray, float t) const;
                __device__ Material *getMat() const;
            private:
                float m_radius;
                Vec3 m_position;
                Material *m_material;
        };
    }
}

#endif