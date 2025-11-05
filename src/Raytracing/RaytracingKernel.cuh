#ifndef RAYTRACING_KERNEL_CUH
#define RAYTRACING_KERNEL_CUH

#include "Vector.cuh"
#include "Ray.cuh"
#include "Sphere.cuh"
#include "Random.cuh"
#include "Material.cuh"
#include <curand_kernel.h>

namespace LR {
    __global__ void renderKernel(cudaSurfaceObject_t surf, Vec3 cameraPos, int width, int height, unsigned long long seed);
}

#endif