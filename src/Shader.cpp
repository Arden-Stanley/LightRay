#include "Shader.h"

namespace LR 
{
	Shader::Use() const
	{
		glUseProgram(m_program);
	}

	unsigned int Shader::m_LoadShader(const std::string &path, Type shaderType) const {
		std::ifstream file(path);
		while (file)
	}

	RenderShader::RenderShader(const std::string &vertexPath, const std::string &fragmentPath) 
	{
				
	}

	RenderShader::~RenderShader() 
	{
			
	}

	RaytracingShader::RaytracingShader(const std::string &computePath)
	{
	
	}
	
	RaytracingShader::~RaytracingShader() 
	{
	
	}
}
