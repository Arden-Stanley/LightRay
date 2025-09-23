#include "Shader.h"

namespace LR 
{
	void Shader::Use() const
	{
		glUseProgram(m_program);
	}

	unsigned int Shader::m_LoadShader(const std::string &path, Type shaderType) const 
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
		case COMPUTE:
			shader = glCreateShader(GL_COMPUTE_SHADER);
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

	void Shader::SetUniform(const std::string &name, int value) const
	{
		int shaderLocation = glGetUniformLocation(m_program, name.c_str());
		glUniform1i(shaderLocation, value);
	}

	RenderShader::RenderShader(const std::string& vertexPath, const std::string& fragmentPath)
	{
		unsigned int vertexShader = m_LoadShader(vertexPath, VERTEX);
		unsigned int fragmentShader = m_LoadShader(fragmentPath, FRAGMENT);

		m_program = glCreateProgram();
		glAttachShader(m_program, vertexShader);
		glAttachShader(m_program, fragmentShader);
		glLinkProgram(m_program);

		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);
	}

	RenderShader::~RenderShader() 
	{
			
	}

	RaytracingShader::RaytracingShader(const std::string &computePath)
	{
		unsigned int computeShader = m_LoadShader(computePath, COMPUTE);

		m_program = glCreateProgram();
		glAttachShader(m_program, computeShader);
		glLinkProgram(m_program);

		glDeleteShader(computeShader);
	}
	
	RaytracingShader::~RaytracingShader() 
	{
	
	}
}
