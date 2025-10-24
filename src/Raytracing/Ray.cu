#include "Ray.cuh"


namespace LR {
    __device__ Ray::Ray(const Vec3& origin, const Vec3& direction) : m_origin(origin), m_dir(direction) {}

    __device__ Vec3 Ray::getOrigin() const {
        return m_origin;
    }

    __device__ Vec3 Ray::getDirection() const {
        return m_dir;
    }

    __device__ Vec3 Ray::pointAt(float t) {
        return m_origin + t * m_dir;
    }

    __device__ Payload::Payload(float t, Vec3 hit, Vec3 normal) 
    : m_t(t), m_hit(hit), m_normal(normal) {}

    __device__ void Payload::setT(float t) {
        m_t = t;
    }
    __device__ void Payload::setHit(const Vec3 &hit) {
        m_hit = hit;
    }

    __device__ void Payload::setNormal(const Vec3 &normal) {
        m_normal = normal;
    }
    
    __device__ float Payload::getT() const {
        return m_t;
    }
    __device__ Vec3 Payload::getHit() const {
        return m_hit;
    }
    __device__ Vec3 Payload::getNormal() const {
        return m_normal;
    }
}