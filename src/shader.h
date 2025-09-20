#ifndef SHADER_H
#define SHADER_H

#include <glad/glad.h>
#include <stdio.h>
#include <stdlib.h>

typedef unsigned int Shader;

Shader create_texture_shader(const char* vertex_path, const char* fragment_path);

Shader* create_raytracing_shader(const char* compute_path);

void use_shader(Shader* shader);

void destroy_shader(Shader* shader);

#endif
