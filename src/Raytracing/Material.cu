#include "Material.cuh"

namespace LR {
    namespace RT {
        __device__ Lambertian::Lambertian(const Vec3 &albedo) : m_albedo(albedo) {}

        __device__ Ray Lambertian::scatter(Ray &ray, Random &randGen) const {
            Vec3 scatterDir = ray.payload.normal + randGen.getVec3();
            return Ray(ray.payload.hit, scatterDir);
        }

        __device__ Vec3 Lambertian::getAlbedo() const {
            return m_albedo;
        }

        __device__ Metal::Metal(const Vec3 &albedo, float fuzz) : m_albedo(albedo), m_fuzz(fuzz) {}

        __device__ Ray Metal::scatter(Ray &ray, Random &randGen) const {
            Vec3 reflected = ray.getDirection() 
            - 2 * dot(ray.getDirection(), ray.payload.normal)
            * ray.payload.normal;
            reflected = unit(reflected) + (m_fuzz * randGen.getVec3());
            return Ray(ray.payload.hit, reflected);
        }

        __device__ Vec3 Metal::getAlbedo() const {
            return m_albedo;
        }

        __device__ Dielectric::Dielectric(float refractionIdx) : m_refraction(refractionIdx) {}

        __device__ Ray Dielectric::scatter(Ray &ray, Random &randGen) const {
            float ri;
            if (dot(ray.getDirection(), ray.payload.normal) > 0.0) {
                ri = 1.0f / m_refraction;
            }
            else {
                ri = m_refraction;
            }


            Vec3 uv = unit(ray.getDirection());
            Vec3 n = ray.payload.normal;

            float cosTheta = fminf(dot(-uv, n), 1.0);
            float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
            bool noRefract = ri * sinTheta > 1.0;
            if (noRefract) {
                Vec3 reflected = ray.getDirection() 
                - 2 * dot(ray.getDirection(), ray.payload.normal)
                * ray.payload.normal;
                reflected = unit(reflected) + (randGen.getVec3());
                return Ray(ray.payload.hit, reflected);
            }
            else {
                Vec3 rayPerp = ri * (uv + cosTheta*n);
                Vec3 rayPara = -sqrt(fabsf(1.0 - rayPerp.lengthSqrd())) * n;
                Vec3 refracted = rayPerp + rayPara;
                return Ray(ray.payload.hit, refracted);
            }
        }

        __device__ Vec3 Dielectric::getAlbedo() const {
            return Vec3(1.0, 1.0, 1.0);
        }
    }
}