#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <iostream>

class vec3 {
public:
    __host__ __device__ vec3();
    __host__ __device__ vec3(float e0, float e1, float e2);  
    __host__ __device__ inline float getX() const {return e[0];}
    __host__ __device__ inline float getY() const {return e[1];}
    __host__ __device__ inline float getZ() const {return e[2];}
    __host__ __device__ inline vec3 operator-() const {return vec3(-e[0], -e[1], -e[2]);}
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) {return e[i];}
    __host__ __device__ inline vec3& operator/=(float t) {return *this *= 1 / t;}
    __host__ __device__ inline float length() const {return std::sqrt(lengthSqrd());}
    __host__ __device__ inline float lengthSqrd() const {return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];}

    __host__ __device__ vec3& operator+=(const vec3& v);
    __host__ __device__ vec3& operator*=(float t);
private:
    float e[3];
};

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];}
__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);}
__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);}
__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);}
__host__ __device__ inline vec3 operator*(float t, const vec3& v) {return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);}
__host__ __device__ inline vec3 operator*(const vec3& v, float t) {return t * v;}
__host__ __device__ inline vec3 operator/(const vec3& v, float t) {return (1/t) * v;}
__host__ __device__ inline float dot(const vec3& u, const vec3& v) {return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];}
__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1], u.e[2] * v.e[0] - u.e[0] * v.e[2], u.e[0] * v.e[1] - u.e[1] * v.e[0]);}
__host__ __device__ inline vec3 unit(const vec3& v) {return v / v.length();}


#endif