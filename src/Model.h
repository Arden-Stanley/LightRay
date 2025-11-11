#ifndef MODEL_H
#define MODEL_H

#include "Common.h"
#include <tiny_gltf.h>

namespace LR
{

    typedef struct {
        glm::vec3 p1, p2, p3;
    } Mesh;

    class Model
    {
    public:
        Model(const std::string &filePath);
        ~Model();
    private:
        std::vector<Mesh> m_primitives;
        tinygltf::Model m_model;
    };
}

#endif