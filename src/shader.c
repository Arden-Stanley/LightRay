#include "shader.h"

Shader* create_texture_shader(const char* vertex_path, const char* fragment_path) {
	FILE* vertex_file = fopen(vertex_path, "w"); 			
	if (!vertex_file) {
		perror("Failed to load vertex file");		
		exit(EXIT_FAILURE);
	}
	fseek(vertex_file, 0, SEEK_END);
	file_length = ftell(vertex_file);
		
}

Shader* create_raytracing_shader(const char* compute_path) {

}

void use_shader(Shader* shader) {

}

void destroy_shader(Shader* shader) {
	
} 
