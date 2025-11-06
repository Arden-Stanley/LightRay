#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <curand_kernel.h>

namespace LR {
        class Vec3 {
            public:
                __host__ __device__ Vec3();
                __host__ __device__ Vec3(float e0, float e1, float e2); 
                __host__ __device__ float X() const;
                __host__ __device__ float Y() const;
                __host__ __device__ float Z() const;
                __host__ __device__ float operator[](int i) const;
                __host__ __device__ float& operator[](int i);

                __host__ __device__ Vec3 operator-() const;
                __host__ __device__ Vec3& operator/=(float t);
                __host__ __device__ float Length() const;
                __host__ __device__ float LengthSqrd() const;
                __host__ __device__ Vec3& operator+=(const Vec3& v);
                __host__ __device__ Vec3& operator*=(float t);
            private:
                float m_vec[3];
        };

        __host__ __device__ Vec3 operator+(const Vec3& u, const Vec3& v);

        __host__ __device__ Vec3 operator-(const Vec3& u, const Vec3& v);

        __host__ __device__ Vec3 operator*(const Vec3& u, const Vec3& v);

        __host__ __device__ Vec3 operator*(float t, const Vec3& v);

        __host__ __device__ Vec3 operator*(const Vec3& v, float t);

        __host__ __device__ Vec3 operator/(const Vec3& v, float t);

        __host__ __device__ float Dot(const Vec3& u, const Vec3& v);

        __host__ __device__ Vec3 Cross(const Vec3& u, const Vec3& v);

        __host__ __device__ Vec3 Unit(const Vec3& v);
}

#endif
