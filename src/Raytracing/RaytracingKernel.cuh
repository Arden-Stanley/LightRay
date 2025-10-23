#ifndef RAYTRACING_KERNEL_CUH
#define RAYTRACING_KERNEL_CUH

namespace LR {
    __global__ void renderKernel(cudaSurfaceObject_t surf, int width, int height);
}

#endif