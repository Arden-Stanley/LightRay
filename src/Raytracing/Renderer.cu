#include "Renderer.h"

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

__device__ bool checkHit(Ray r) {
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

__global__ void renderKernel(cudaSurfaceObject_t *surf, int width, int height) {
    

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) {
        return;
    }

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
    
    surf2Dwrite(pixelColor, *surf, i * sizeof(float4), j);
}

Renderer::Renderer(unsigned int texHandle, int screenWidth, int screenHeight) 
: m_texHandle(texHandle), m_screenWidth(screenWidth), m_screenHeight(screenHeight) {}

Renderer::~Renderer() {
    //cudaGraphicsUnregisterResource(m_texPtr);
}

void Renderer::render() {
    cudaGraphicsResource_t m_texPtr;
    cudaArray_t m_mappedTex;
    cudaGraphicsGLRegisterImage(&m_texPtr, m_texHandle, GL_TEXTURE_2D, NULL);
    cudaGraphicsMapResources(1, &m_texPtr, 0);
    cudaGraphicsSubResourceGetMappedArray(&m_mappedTex, m_texPtr, 0, 0);

    cudaResourceDesc resDesc; 
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_mappedTex;

    cudaSurfaceObject_t surf;
    cudaCreateSurfaceObject(&surf, &resDesc);

    int tx = 8;
    int ty = 8;

    dim3 blocks(m_screenWidth / tx + 1, m_screenHeight / ty, + 1); 
    dim3 threads(tx, ty);
    renderKernel<<<blocks, threads>>>(&surf, m_screenWidth, m_screenHeight);
    cudaDeviceSynchronize();

    cudaDestroySurfaceObject(surf);
    cudaGraphicsUnmapResources(1, &m_texPtr, 0);
}