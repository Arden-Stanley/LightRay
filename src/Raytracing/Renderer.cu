#include "Renderer.h"

#include "RaytracingKernel.cuh"
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


namespace LR {
    static cudaGraphicsResource *m_texPtr;
    static cudaArray_t m_mappedTex;
    static cudaSurfaceObject_t surf;

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

    void Renderer::Render(const Pos &camPos) {
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

        dim3 threads(64, 64);
        dim3 blocks((m_screenWidth + threads.x - 1) / threads.x, (m_screenHeight + threads.y - 1) / threads.y); 

        static float rando = 10;
        RenderKernel<<<threads, blocks>>>(surf, {camPos.x, camPos.y, camPos.z},m_screenWidth, m_screenHeight, 100);
        rando += 21;
        
        
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
}