#include "Shader.h"

namespace LR 
{
	Shader::Shader(const std::string& vertexPath, const std::string& fragmentPath)
	{
		unsigned int vertexShader = m_loadShader(vertexPath, VERTEX);
		unsigned int fragmentShader = m_loadShader(fragmentPath, FRAGMENT);

		m_program = glCreateProgram();
		glAttachShader(m_program, vertexShader);
		glAttachShader(m_program, fragmentShader);
		glLinkProgram(m_program);

		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);
	}
	
	Shader::~Shader() {}

	unsigned int Shader::m_loadShader(const std::string &path, Type shaderType) const 
	{
		std::ifstream file(path);
		if (!file)
		{
			std::cerr << "Failed to open file!" << std::endl;
		}
		std::string line;
		std::string sourceCode = "";
		while (std::getline(file, line)) 
		{
			sourceCode += line;
			sourceCode += '\n';
		}

		sourceCode += '\0';

		//std::cout << sourceCode << std::endl;

		file.close();

		unsigned int shader = 0;
		switch (shaderType) 
		{
		case VERTEX:
			shader = glCreateShader(GL_VERTEX_SHADER);
			break;
		case FRAGMENT:
			shader = glCreateShader(GL_FRAGMENT_SHADER);
			break;
		default: 
			break;
		}

		const char* rawSource = sourceCode.c_str();
		glShaderSource(shader, 1, &rawSource, NULL);
		glCompileShader(shader);

		int success;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success) 
		{
			char message[1024];
			glGetShaderInfoLog(shader, 1024, NULL, message);
			std::cerr << "Error compiling shader:" << message << std::endl;
		}

		return shader;
	}

	void Shader::use() const {
		glUseProgram(m_program);
	}
}
