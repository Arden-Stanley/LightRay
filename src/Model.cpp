#include "Model.h"

namespace LR
{
    Model::Model(const std::string &filePath) : m_primitives(), m_model() {
        tinygltf::TinyGLTF loader;

        loader.LoadBinaryFromFile(&m_model, nullptr, nullptr, filePath.c_str());

        for (const auto &mesh : m_model.meshes) {
            for (const auto &primitive : mesh.primitives) {
                
                if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                    continue;
                }

                const tinygltf::Accessor &accessor = m_model.accessors[primitive.indices];
                const tinygltf::BufferView &bufferView = m_model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer &buffer = m_model.buffers[bufferView.buffer];

                const unsigned char* data = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
                std::vector<glm::vec3> temp = {};
                for (size_t i = 0; i < accessor.count; i++) {
                    const float *vertexData = reinterpret_cast<const float *>(data + i * accessor.ByteStride(bufferView));
                    glm::vec3 pos = glm::vec3{vertexData[0], vertexData[1], vertexData[2]};
                    temp.push_back(pos);
                }
                m_primitives.push_back(Mesh{temp[0], temp[1], temp[2]});
            }
        }
    }

    Model::~Model() {}
}
