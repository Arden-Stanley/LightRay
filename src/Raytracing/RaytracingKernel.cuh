#ifndef RAYTRACING_KERNEL_CUH
#define RAYTRACING_KERNEL_CUH

#include "Ray.cuh"
#include "Vector.cuh"
#include "Sphere.cuh"
#include "Random.cuh"
#include <curand_kernel.h>

namespace LR {
    __global__ void renderKernel(cudaSurfaceObject_t surf, int width, int height, unsigned long long seed);
}

#endif