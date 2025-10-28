#include "Vector.cuh" 


namespace LR {
    namespace RT {
        __host__ __device__ Vec3::Vec3() : m_vec{0, 0, 0} {}

        __host__ __device__ Vec3::Vec3(float x, float y, float z) : m_vec{x, y, z} {}  

        __host__ __device__ float Vec3::getX() const {
            return m_vec[0];
        }

        __host__ __device__ float Vec3::getY() const {
            return m_vec[1];
        }

        __host__ __device__ float Vec3::getZ() const {
            return m_vec[2];
        }

        __host__ __device__ float Vec3::operator[](int i) const {
            return m_vec[i];
        }

        __host__ __device__ float& Vec3::operator[](int i) {
            return m_vec[i];
        }

        __host__ __device__ Vec3 Vec3::operator-() const {
            return Vec3(-m_vec[0], -m_vec[1], -m_vec[2]);
        }

        __host__ __device__ Vec3& Vec3::operator/=(float t) {
            return *this *= 1 / t;
        }

        __host__ __device__ float Vec3::length() const {
            return std::sqrt(lengthSqrd());
        }

        __host__ __device__ float Vec3::lengthSqrd() const {
            return m_vec[0] * m_vec[0] + m_vec[1] * m_vec[1] + m_vec[2] * m_vec[2];
        }

        __host__ __device__ Vec3& Vec3::operator+=(const Vec3& v) {
            m_vec[0] += v.m_vec[0]; 
            m_vec[1] += v.m_vec[1]; 
            m_vec[2] += v.m_vec[2]; 
            return *this;
        }

        __host__ __device__ Vec3& Vec3::operator*=(float t) {
            m_vec[0] *= t; 
            m_vec[1] *= t; 
            m_vec[2] *= t; 
            return *this;
        }

        __host__ __device__ Vec3 operator+(const Vec3& u, const Vec3& v) {
            return Vec3(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
        }

        __host__ __device__ Vec3 operator-(const Vec3& u, const Vec3& v) {
            return Vec3(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
        }

        __host__ __device__ Vec3 operator*(const Vec3& u, const Vec3& v) {
            return Vec3(u[0] * v[0], u[1] * v[1], u[2] * v[2]);
        }

        __host__ __device__ Vec3 operator*(float t, const Vec3& v) {
            return Vec3(t*v[0], t*v[1], t*v[2]);
        }

        __host__ __device__ Vec3 operator*(const Vec3& v, float t) {
            return t * v;
        }

        __host__ __device__ Vec3 operator/(const Vec3& v, float t) {
            return (1/t) * v;
        }

        __host__ __device__ float dot(const Vec3& u, const Vec3& v) {
            return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        }

        __host__ __device__ Vec3 cross(const Vec3& u, const Vec3& v) {
            return Vec3(
                u[1] * v[2] - u[2] * v[1], 
                u[2] * v[0] - u[0] * v[2], 
                u[0] * v[1] - u[1] * v[0]
            );
        }

        __host__ __device__ Vec3 unit(const Vec3& v) {
            return v / v.length();
        }
    }
}