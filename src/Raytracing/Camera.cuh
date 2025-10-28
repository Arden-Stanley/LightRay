#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "Vector.cuh"

namespace LR {
    namespace RT {
        class Camera {
            public:
                __host__ __device__ Camera();
                
            private:
                Vec3 m_pos;
        };
    }
}

#endif