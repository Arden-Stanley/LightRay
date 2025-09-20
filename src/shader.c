#include "shader.h"

Shader create_texture_shader(const char* vertex_path, const char* fragment_path) {
	FILE* vertex_file = fopen(vertex_path, "w"); 			
	if (!vertex_file) {
		perror("Failed to load vertex file");		
		exit(EXIT_FAILURE);
	}
	fseek(vertex_file, 0, SEEK_END);
	int file_length = ftell(vertex_file);
	rewind(vertex_file);			
	const char* vertex_src = (char *)malloc(file_length * sizeof(char));
	
	fread(vertex_src, sizeof(char), file_length, vertex_file);
	fclose(vertex_file);

	Shader shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(shader, 1, vertex_src, NULL);
	glCompileShader(shader);

}

Shader* create_raytracing_shader(const char* compute_path) {

}

void use_shader(Shader* shader) {

}

void destroy_shader(Shader* shader) {
	
} 
