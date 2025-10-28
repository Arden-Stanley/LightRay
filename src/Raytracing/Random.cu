#include "Random.cuh"

namespace LR {
    namespace RT {
        __device__ Random::Random(curandState &state) : m_state(state) {}

        __device__ Vec3 Random::getVec3() {
            Vec3 v;
            while (true) {
                v[0] = curand_uniform(&m_state) * 2.0f - 1.0f;
                v[1] = curand_uniform(&m_state) * 2.0f - 1.0f;
                v[2] = curand_uniform(&m_state) * 2.0f - 1.0f;
                float lensq = v.lengthSqrd();
                if (1e-160 < lensq && lensq <= 1.0f) {
                    v = v / sqrt(lensq);
                    break;
                }
            }
            return v;
            /*
            if (dot(v, normal) > 0.0f) {
                return v;
            }
            else {
                return -v;
            }
            */
        }

        __device__ Ray Random::getSampRay(int i, int j, Vec3 p00, Vec3 du, Vec3 dv, Vec3 center) {
            Vec3 v;
            v[0] = curand_uniform(&m_state) - 0.5f;
            v[1] = curand_uniform(&m_state) - 0.5f;
            v[2] = 0.0;

            Vec3 sample = p00 + ((i + v.getX()) * du) + ((j + v.getY()) * dv);
            return Ray(center, sample - center);
        }
    }

}