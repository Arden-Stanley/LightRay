#include "Renderer.h"

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "Vector.cuh"
#include "Ray.cuh"


__device__ bool checkHit(LR::Ray r) {
    Vec3 center = Vec3(0.0, 0.0, -3.0);
    float radius = 1.0f;
    Vec3 oc = center - r.getOrigin();
    float a = dot(r.getDirection(), r.getDirection());
    float b = -2.0f * dot(r.getDirection(), oc);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return false;
    }
    else {
        return true;
    }
}

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
        float4 pixelColor = make_float4(0.0, 0.0, 0.0, 1.0);
        if (checkHit(ray)) {
            pixelColor = make_float4(1.0, 0.0, 0.0, 1.0);
        }
        surf2Dwrite(pixelColor, surf, i * sizeof(float4), j);
    }    
}


cudaGraphicsResource *m_texPtr;
cudaArray_t m_mappedTex;
cudaSurfaceObject_t surf;

Renderer::Renderer(unsigned int texHandle, int screenWidth, int screenHeight) 
: m_texHandle(texHandle), m_screenWidth(screenWidth), m_screenHeight(screenHeight) {

    cudaGraphicsGLRegisterImage(&m_texPtr, m_texHandle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error 1: " << cudaGetErrorString(err) << "\n";
    }

}

Renderer::~Renderer() {
    cudaGraphicsUnregisterResource(m_texPtr);
}

void Renderer::render() {
    cudaGraphicsMapResources(1, &m_texPtr, 0);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error 2: " << cudaGetErrorString(err) << "\n";
    }
    cudaGraphicsSubResourceGetMappedArray(&m_mappedTex, m_texPtr, 0, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error 3: " << cudaGetErrorString(err) << "\n";
    }



    cudaResourceDesc resDesc = {};
    //memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_mappedTex;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error 4: " << cudaGetErrorString(err) << "\n";
    }

    
    cudaCreateSurfaceObject(&surf, &resDesc);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error 5: " << cudaGetErrorString(err) << "\n";
    }

    
    dim3 blocks(16, 16); 
    dim3 grid((m_screenWidth + 15) / 16, (m_screenHeight + 15) / 16);

    renderKernel<<<grid, blocks>>>(surf, m_screenWidth, m_screenHeight);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error 6: " << cudaGetErrorString(err) << "\n";
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error 7: " << cudaGetErrorString(err) << "\n";
    }

    cudaDestroySurfaceObject(surf);
    cudaGraphicsUnmapResources(1, &m_texPtr, 0);

}