#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "Ray.cuh"
#include "Vector.cuh"
#include "Random.cuh"

namespace LR {
    class Material {
        public:
            virtual ~Material() = default;
            __device__ virtual Ray scatter(Ray &ray, Random &randGen) const = 0;
            __device__ virtual Vec3 getAlbedo() const = 0;
    };

    class Lambertian : public Material {
        public:
            __device__ Lambertian(const Vec3 &albedo);
            __device__ virtual Ray scatter(Ray &ray, Random &randGen) const override;
            __device__ virtual Vec3 getAlbedo() const override;
        private:
            Vec3 m_albedo;
    };

    class Metal : public Material {
        public:
            __device__ Metal(const Vec3 &albedo, float fuzz);
            __device__ virtual Ray scatter(Ray &ray, Random &randgen) const override;
            __device__ virtual Vec3 getAlbedo() const override;
        private:
            Vec3 m_albedo;
            float m_fuzz;
    };

    class Dielectric : public Material {
        public:
            __device__ Dielectric(float refractionIdx);
            __device__ virtual Ray scatter(Ray &ray, Random &randGen) const override;
            __device__ virtual Vec3 getAlbedo() const override;
        private:
            double m_refraction;
    };
}

#endif