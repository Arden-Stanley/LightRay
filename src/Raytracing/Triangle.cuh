#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "Material.cuh"
#include "Ray.cuh"
#include "Vector.cuh"

namespace LR {
    class Triangle {
        public:
            __device__ Triangle(const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, Material *material);
            __device__ ~Triangle();
            __device__ bool CheckHit(Ray &ray) const;
            __device__ Material* Mat() const;
        private:
            __device__ void Payload(Ray &ray, float t) const;
        private:
            Vec3 m_p1, m_p2, m_p3;
            Vec3 m_normal;
            Material *m_material;
    };
}

#endif