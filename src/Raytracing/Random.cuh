#ifndef RANDOM_CUH
#define RANDOM_CUH

#include <curand_kernel.h>
#include "Vector.cuh"
#include "Ray.cuh"

namespace LR {
    namespace RT { 
        class Random {
            public:
                __device__ Random(curandState &state);
                __device__ Vec3 getVec3();
                __device__ Ray getSampRay(int i, int j, Vec3 p00, Vec3 du, Vec3 dv, Vec3 center);
            private:
                curandState m_state;
        };
    }         
}

#endif