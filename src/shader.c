#include "shader.h"

static Shader load_shader(const char* path, Shader_Type type) {
	FILE* file = fopen(path, "r"); 			
	if (!file) {
		perror("Failed to load shader file");		
		exit(EXIT_FAILURE);
	}
	fseek(file, 0, SEEK_END);
	int file_length = ftell(file) + 1;
	rewind(file);		
	
	char* src = (char *)malloc(file_length * sizeof(char));

	fread(src, sizeof(char), file_length - 1, file);
	src[file_length] = '\0';

	fclose(file);
	
	Shader shader;	
	switch (type) {	
		case VERTEX:
			shader = glCreateShader(GL_VERTEX_SHADER);	
		case FRAGMENT:
			shader = glCreateShader(GL_FRAGMENT_SHADER);
	}

	const char* src_code = src;	
	glShaderSource(shader, 1, &src_code, NULL);	
	
	glCompileShader(shader);
	int success;
	char info_log[512];
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(shader, 512, NULL, info_log);	
		printf("Failed to compile shader: %s", info_log);
		exit(EXIT_FAILURE);
	}
	
	free(src);
	return shader;	
}

Shader create_texture_shader(const char* vertex_path, const char* fragment_path) {
	Shader vertex_shader = load_shader(vertex_path, VERTEX);		
	Shader fragment_shader = load_shader(fragment_path, FRAGMENT);
	Shader program = glCreateProgram();
	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);
	glLinkProgram(program);	
	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);
	return program;
}

Shader* create_raytracing_shader(const char* compute_path) {
	
}

void use_shader(Shader shader) {
	glUseProgram(shader);
}

void destroy_shader(Shader shader) {
	glDeleteProgram(shader);	
} 
