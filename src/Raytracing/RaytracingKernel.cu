#include "RaytracingKernel.cuh"

#include "Ray.cuh"
#include "Vector.cuh"
#include "Sphere.cuh"

namespace LR{
    __global__ void renderKernel(cudaSurfaceObject_t surf, int width, int height) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if ((i < width) && (j < height)) {
            float focalLength = 1.0;
            Vec3 cameraCenter(0, 0, 0);
            float viewportHeight = 2.0;
            float viewportWidth = viewportHeight * (float(width) / height);
            Vec3 u = Vec3(viewportWidth, 0, 0);
            Vec3 v = Vec3(0, -viewportHeight, 0);
            Vec3 du = u / float(width);
            Vec3 dv = v / float(height);
            Vec3 upperLeft = cameraCenter - Vec3(0, 0, focalLength) - (u / 2) - (v / 2);
            Vec3 firstPixel = upperLeft + 0.5f * (du + dv);
            Vec3 targetPixel = firstPixel + (i * du) + (j * dv);
        
            Ray ray(cameraCenter, targetPixel - cameraCenter);

            Sphere sphere(1.0, {0.0, 0.0, -2.0});
            float4 pixelColor = make_float4(0.0, 0.0, 0.0, 1.0);
            if (sphere.checkHit(ray)) {
                pixelColor = make_float4(1.0, 0.0, 0.0, 1.0);
            }
            surf2Dwrite(pixelColor, surf, i * sizeof(float4), j);
        }    
    }
}