#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <curand_kernel.h>

namespace LR {
    namespace RT {
        class Vec3 {
            public:
                __host__ __device__ Vec3();
                __host__ __device__ Vec3(float e0, float e1, float e2); 
                __host__ __device__ float getX() const;
                __host__ __device__ float getY() const;
                __host__ __device__ float getZ() const;
                __host__ __device__ float operator[](int i) const;
                __host__ __device__ float& operator[](int i);

                __host__ __device__ Vec3 operator-() const;
                __host__ __device__ Vec3& operator/=(float t);
                __host__ __device__ float length() const;
                __host__ __device__ float lengthSqrd() const;
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

        __host__ __device__ float dot(const Vec3& u, const Vec3& v);

        __host__ __device__ Vec3 cross(const Vec3& u, const Vec3& v);

        __host__ __device__ Vec3 unit(const Vec3& v);
    }
}

#endif
