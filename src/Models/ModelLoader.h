#pragma once
#include <string>
#include <tiny_gltf.h>
#include "Model.h"

namespace LR
{
    class ModelLoader {
    public:
        static bool LoadModel(const std::string& filepath, Model& model);
    };
}