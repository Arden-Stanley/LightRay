#pragma once
#include <vector>
#include <string>
#include <glad/glad.h>
#include <glm/glm.hpp>

namespace LR
{
    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
    };

    class Model
    {
    public:
        Model() = default;
        ~Model();

        bool LoadFromFile(const std::string& filepath);
        void Draw() const;

    private:
        GLuint VAO = 0, VBO = 0, EBO = 0;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        void SetupBuffers();
        friend class ModelLoader; // ModelLoader can access vertices/indices directly
    };
}