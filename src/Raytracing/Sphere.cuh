#ifndef SPHERE_H
#define SPHERE_H

#include "Vector.cuh"
#include "Ray.cuh"
#include "Material.cuh"
#include <cuda_runtime.h>

namespace LR {

        class Sphere {
            public:
                __device__ Sphere(float r, const Vec3& pos, Material *material);
                __device__ ~Sphere();
                __device__ bool CheckHit(Ray &ray) const;
                __device__ Vec3 Center() const;
                __device__ float Radius() const;
                __device__ Material* Mat() const;
            private:
                __device__ void Payload(Ray &ray, float t) const;
            private:
                float m_radius;
                Vec3 m_position;
                Material *m_material;
        };
}

#endif