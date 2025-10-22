#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>

class Vec3 {
    public:
        float e[3];

        __host__ __device__ Vec3() : e{0.0, 0.0, 0.0} {}
        __host__ __device__ Vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}  
        __host__ __device__ inline float getX() const {return e[0];}
        __host__ __device__ inline float getY() const {return e[1];}
        __host__ __device__ inline float getZ() const {return e[2];}
        __host__ __device__ inline float operator[](int i) const { return e[i]; }
        __host__ __device__ inline float& operator[](int i) {return e[i];}

        __host__ __device__ inline Vec3 operator-() const {return Vec3(-e[0], -e[1], -e[2]);}
        __host__ __device__ inline Vec3& operator/=(float t) {return *this *= 1 / t;}
        __host__ __device__ inline float length() const {return std::sqrt(lengthSqrd());}
        __host__ __device__ inline float lengthSqrd() const {return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];}
        __host__ __device__ inline Vec3& operator+=(const Vec3& v) {e[0] += v.e[0]; e[1] += v.e[1]; e[2] += v.e[2]; return *this;}
        __host__ __device__ inline Vec3& operator*=(float t) {e[0] *= t; e[1] *= t; e[2] *= t; return *this;}
};

__host__ __device__ inline Vec3 operator+(const Vec3& u, const Vec3& v) {
    return Vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline Vec3 operator-(const Vec3& u, const Vec3& v) {
    return Vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3& u, const Vec3& v) {
    return Vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) {
    return Vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3& v, float t) {return t * v;}

__host__ __device__ inline Vec3 operator/(const Vec3& v, float t) {return (1/t) * v;}

__host__ __device__ inline float dot(const Vec3& u, const Vec3& v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(
        u.e[1] * v.e[2] - u.e[2] * v.e[1], 
        u.e[2] * v.e[0] - u.e[0] * v.e[2], 
        u.e[0] * v.e[1] - u.e[1] * v.e[0]
    );
}

__host__ __device__ inline Vec3 unit(const Vec3& v) {return v / v.length();}

inline std::ostream& operator<<(std::ostream &os, const Vec3 &t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2]; 
    return os;
}


class Ray {
    public:
        __device__ Ray() {}
        __device__ Ray(const Vec3& origin, const Vec3& direction) : m_origin(origin), m_dir(direction) {}
        __device__ inline Vec3 getOrigin() const {return m_origin;}
        __device__ inline Vec3 getDirection() const {return m_dir;}
        __device__ inline Vec3 pointAt(float t) {return m_origin + t * m_dir;}
    private:
        Vec3 m_origin;
        Vec3 m_dir;
};

#endif