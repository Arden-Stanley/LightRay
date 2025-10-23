#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"
#include "ModelLoader.h"
#include "Model.h"
#include <iostream>


namespace LR
{
    bool ModelLoader::LoadModel(const std::string& filepath, Model& model)
    {
        tinygltf::TinyGLTF loader;
        tinygltf::Model gltfModel;
        std::string err, warn;

        bool ret = loader.LoadASCIIFromFile(&gltfModel, &err, &warn, filepath);
        if (!warn.empty()) std::cout << "glTF Warning: " << warn << std::endl;
        if (!err.empty()) std::cerr << "glTF Error: " << err << std::endl;
        if (!ret) 
        {
            std::cerr << "Failed to load glTF: " << filepath << std::endl;
            return false;
        }

        if (gltfModel.meshes.empty()) 
        {
            std::cerr << "No meshes found in model." << std::endl;
            return false;
        }

        const tinygltf::Mesh& mesh = gltfModel.meshes[0];
        const tinygltf::Primitive& prim = mesh.primitives[0];

        const auto& posAccessor = gltfModel.accessors[prim.attributes.at("POSITION")];
        const auto& posBufferView = gltfModel.bufferViews[posAccessor.bufferView];
        const auto& posBuffer = gltfModel.buffers[posBufferView.buffer];

        const float* posData = reinterpret_cast<const float*>(
            posBuffer.data.data() + posBufferView.byteOffset + posAccessor.byteOffset
            );

        model.vertices.resize(posAccessor.count);

        for (size_t i = 0; i < posAccessor.count; i++) {
            model.vertices[i].position = glm::vec3(
                posData[i * 3 + 0],
                posData[i * 3 + 1],
                posData[i * 3 + 2]
            );
            model.vertices[i].normal = glm::vec3(0.0f); // No normals for now
        }

        if (prim.indices > -1) {
            const auto& idxAccessor = gltfModel.accessors[prim.indices];
            const auto& idxBufferView = gltfModel.bufferViews[idxAccessor.bufferView];
            const auto& idxBuffer = gltfModel.buffers[idxBufferView.buffer];

            const uint16_t* indexData = reinterpret_cast<const uint16_t*>(
                idxBuffer.data.data() + idxBufferView.byteOffset + idxAccessor.byteOffset
                );

            model.indices.resize(idxAccessor.count);
            for (size_t i = 0; i < idxAccessor.count; ++i) {
                model.indices[i] = static_cast<uint32_t>(indexData[i]);
            }
        }

        model.SetupBuffers();
        return true;
    }
}
