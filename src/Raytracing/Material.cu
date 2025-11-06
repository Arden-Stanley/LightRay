#include "Material.cuh"

namespace LR {
        __device__ Lambertian::Lambertian(const Vec3 &albedo) : m_albedo(albedo) {}

        __device__ Ray Lambertian::Scatter(Ray &ray, Random &randGen) const {
            Vec3 scatterDir = ray.payload.normal + randGen.RandVec();
            return Ray(ray.payload.hit, scatterDir);
        }

        __device__ Vec3 Lambertian::Albedo() const {
            return m_albedo;
        }

        __device__ Metal::Metal(const Vec3 &albedo, float fuzz) : m_albedo(albedo), m_fuzz(fuzz) {}

        __device__ Ray Metal::Scatter(Ray &ray, Random &randGen) const {
            Vec3 reflected = ray.Direction() 
            - 2 * Dot(ray.Direction(), ray.payload.normal)
            * ray.payload.normal;
            reflected = Unit(reflected) + (m_fuzz * randGen.RandVec());
            return Ray(ray.payload.hit, reflected);
        }

        __device__ Vec3 Metal::Albedo() const {
            return m_albedo;
        }

        __device__ Dielectric::Dielectric(float refractionIdx) : m_refraction(refractionIdx) {}

        __device__ Ray Dielectric::Scatter(Ray &ray, Random &randGen) const {
            float ri;
            if (Dot(ray.Direction(), ray.payload.normal) > 0.0) {
                ri = 1.0f / m_refraction;
            }
            else {
                ri = m_refraction;
            }


            Vec3 uv = Unit(ray.Direction());
            Vec3 n = ray.payload.normal;

            float cosTheta = fminf(Dot(-uv, n), 1.0);
            float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
            bool noRefract = ri * sinTheta > 1.0;
            if (noRefract) {
                Vec3 reflected = ray.Direction() 
                - 2 * Dot(ray.Direction(), ray.payload.normal)
                * ray.payload.normal;
                reflected = Unit(reflected) + (randGen.RandVec());
                return Ray(ray.payload.hit, reflected);
            }
            else {
                Vec3 rayPerp = ri * (uv + cosTheta*n);
                Vec3 rayPara = -sqrt(fabsf(1.0 - rayPerp.LengthSqrd())) * n;
                Vec3 refracted = rayPerp + rayPara;
                return Ray(ray.payload.hit, refracted);
            }
        }

        __device__ Vec3 Dielectric::Albedo() const {
            return Vec3(1.0, 1.0, 1.0);
        }
}