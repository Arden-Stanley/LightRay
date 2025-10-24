#include "RaytracingKernel.cuh"

namespace LR{
    __global__ void renderKernel(cudaSurfaceObject_t surf, int width, int height, unsigned long long seed) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if ((i < width) && (j < height)) {
            curandState state;
            curand_init(seed, i * j, 0, &state);
            Random randGen(state);

            
            float focalLength = 1.0;
            Vec3 cameraCenter(0, 0, 0);
            float viewportHeight = 2.0;
            float viewportWidth = viewportHeight * (float(width) / height);
            Vec3 u = Vec3(viewportWidth, 0, 0);
            Vec3 v = Vec3(0, viewportHeight, 0);
            Vec3 du = u / float(width);
            Vec3 dv = v / float(height);
            Vec3 upperLeft = cameraCenter - Vec3(0, 0, focalLength) - (u / 2) - (v / 2);
            Vec3 firstPixel = upperLeft + 0.5f * (du + dv);

            //Vec3 targetPixel = firstPixel + (i * du) + (j * dv);
            
            Sphere sphere(1.0, {0.0, 0.0, -3.0});
            Sphere ground(100.0, {0.0, -101.0, -3.0});

            Vec3 finalColor = Vec3(0, 0, 0);
            for (int s = 0; s < 50; s++) {
                Ray ray = randGen.getSampRay(i, j, firstPixel, du, dv, cameraCenter);
                float attenuation = 1.0;
                Vec3 color(0.5, 0.8, 1.0);
                for (int idx = 0; idx < 5; idx++) {
                    if (sphere.checkHit(ray)) {
                        attenuation *= 0.6;
                    }
                    else if (ground.checkHit(ray)) {
                        attenuation *= 0.6;
                    }
                    else { 
                        finalColor += (color * attenuation);
                        break;
                    }
                    Vec3 targ = ray.payload.getNormal() + randGen.getVec3(ray.payload.getNormal());
                    ray = Ray(ray.payload.getHit(), targ);
                }
            }

            finalColor = finalColor / 50;
            float4 pixelColor = make_float4(finalColor.getX(), finalColor.getY(), finalColor.getZ(), 1.0);
            surf2Dwrite(pixelColor, surf, i * sizeof(float4), j);
        }    
    }
}