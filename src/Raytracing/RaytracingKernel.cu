#include "RaytracingKernel.cuh"

namespace LR{
    __global__ void renderKernel(cudaSurfaceObject_t surf, int width, int height, unsigned long long seed) {
        using namespace RT;

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
            
            Lambertian mat1({1.0, 0.0, 1.0});
            Lambertian mat2({0.0, 1.0, 0.2});
            Metal mat3({0.3, 0.3, 0.3}, 0.05);
            Dielectric mat4(0.9);

            Sphere sphere(1.0, {-2.0, 0.0, -3.0}, &mat1);
            Sphere mirror(1.0, {2.0, 0.0, -3.0}, &mat3);
            Sphere glass(1.0, {0.0, 0.0, -3.0}, &mat4);
            Sphere ground(100.0, {0.0, -101.0, -3.0}, &mat2);

            Vec3 finalColor = Vec3(0, 0, 0);
            for (int s = 0; s < 40; s++) {
                Ray ray = randGen.getSampRay(i, j, firstPixel, du, dv, cameraCenter);
                Vec3 color(0.5, 0.8, 1.0);
                Material *mat;
                for (int idx = 0; idx < 10; idx++) {
                    if (sphere.checkHit(ray)) {
                        mat = sphere.getMat();
                    }
                    else if (mirror.checkHit(ray)) {
                        mat = mirror.getMat();
                    }
                    //else if (glass.checkHit(ray)) {
                    //    mat = glass.getMat();
                    //}
                    else if (ground.checkHit(ray)) {
                        mat = ground.getMat();
                    }
                    else { 
                        break;
                    }
                    color = color * mat->getAlbedo();
                    ray = mat->scatter(ray, randGen);
                }
                finalColor += color;
            }

            finalColor = finalColor / 40;
            float4 pixelColor = make_float4(finalColor.getX(), finalColor.getY(), finalColor.getZ(), 1.0);
            surf2Dwrite(pixelColor, surf, i * sizeof(float4), j);
        }    
    }
}