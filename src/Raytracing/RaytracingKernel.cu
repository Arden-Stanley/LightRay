#include "RaytracingKernel.cuh"

namespace LR {
    __global__ void RenderKernel(cudaSurfaceObject_t surf, Vec3 cameraPos, int width, int height, unsigned long long seed) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if ((i < width) && (j < height)) {
            curandState state;
            curand_init(seed, i * j, 0, &state);
            Random randGen(state);

            
            float focalLength = 1.0;
            Vec3 cameraCenter = cameraPos;
            float viewportHeight = 2.0;
            float viewportWidth = viewportHeight * (float(width) / height);
            Vec3 u = Vec3(viewportWidth, 0, 0);
            Vec3 v = Vec3(0, viewportHeight, 0);
            Vec3 du = u / float(width);
            Vec3 dv = v / float(height);
            Vec3 upperLeft = cameraCenter - Vec3(0, 0, focalLength) - (u / 2) - (v / 2);
            Vec3 firstPixel = upperLeft + 0.5f * (du + dv);
            
            Lambertian mat1({1.0, 0.0, 1.0});
            Lambertian mat2({0.0, 1.0, 0.2});
            Metal mat3({0.8, 0.8, 0.8}, 0.05);
            Dielectric mat4(0.9);
            Metal mat5({0.8, 0.6, 0.6}, 0.6);

            Sphere sphere(1.0, {-2.0, 0.0, -3.0}, &mat1);
            Sphere mirror(1.0, {2.0, 0.0, -3.0}, &mat3);
            Sphere metal(0.7, {0.0, 0.0, -3.0}, &mat5);
            Sphere glass(1.0, {0.0, 0.0, -3.0}, &mat4);
            Sphere ground(100.0, {0.0, -101.0, -3.0}, &mat2);

            Vec3 finalColor = Vec3(0, 0, 0);
            for (int s = 0; s < 5; s++) {
                Ray ray = randGen.SampRay(i, j, firstPixel, du, dv, cameraCenter);
                Vec3 color(0.5, 0.8, 1.0);
                Material *mat;
                for (int idx = 0; idx < 5; idx++) {
                    if (sphere.CheckHit(ray)) {
                        mat = sphere.Mat();
                    }
                    
                    else if (mirror.CheckHit(ray)) {
                        mat = mirror.Mat();
                    }
                    
                    else if (metal.CheckHit(ray)) {
                        mat = metal.Mat();
                    }
                    //else if (glass.checkHit(ray)) {
                       // mat = glass.getMat();
                   // }
                    else if (ground.CheckHit(ray)) {
                        mat = ground.Mat();
                    }
                    else { 
                        break;
                    }
                    color = color * mat->Albedo();
                    ray = mat->Scatter(ray, randGen);
                }
                finalColor += color;
            }

            finalColor = finalColor / 5;
            float4 pixelColor = make_float4(finalColor.X(), finalColor.Y(), finalColor.Z(), 1.0);
            surf2Dwrite(pixelColor, surf, i * sizeof(float4), j);
        }    
    }
}