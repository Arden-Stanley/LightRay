#include "Triangle.cuh"

namespace LR {
    __device__ Triangle::Triangle(const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, Material *material) 
    : m_p1(p1), m_p2(p2), m_p3(p3), m_normal(), m_material(material) {
        Vec3 edge1 = m_p2 - m_p1;
        Vec3 edge2 = m_p3 - m_p1;
        m_normal = Cross(edge1, edge2);
    }

    __device__ Triangle::~Triangle() {}

    __device__ bool Triangle::CheckHit(Ray &ray) const {
        //Moller-Trumbore intersection
        Vec3 edge1 = m_p2 - m_p1;
        Vec3 edge2 = m_p3 - m_p1;
        Vec3 rayE2 = Cross(ray.Direction(), edge2);
        float determinant = Dot(edge1, rayE2);

        float inverseDet = 1.0 / determinant;
        Vec3 s = ray.Origin() - m_p1;
        float u = inverseDet * Dot(s, rayE2);
        if (u < 0.0 || u > 1) {
            return false;
        }
        
        Vec3 sE1 = Cross(s, edge1);
        float v = inverseDet * Dot(ray.Direction(), sE1);

        if (v < 0.0 || u + v > 1) {
            return false;
        }

        float t = inverseDet * Dot(edge2, sE1);
        
        if (t < 0.0) {
            return false;
        }

        Payload(ray, t);
        return true;
    }

    __device__ void Triangle::Payload(Ray &ray, float t) const {
        Vec3 hit = ray.PointAt(t);
        ray.payload.t = t;
        ray.payload.hit = hit;
        ray.payload.normal = m_normal;
    }

    __device__ Material* Triangle::Mat() const {
        return m_material;
    }
}