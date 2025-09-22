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
		
		unsigned int shader;
		return shader;
	}

	RenderShader::RenderShader(const std::string& vertexPath, const std::string& fragmentPath)
		: m_program(NULL)
	{
		
	}

	RenderShader::~RenderShader() 
	{
			
	}

	RaytracingShader::RaytracingShader(const std::string &computePath)
		: m_program(NULL)
	{
	
	}
	
	RaytracingShader::~RaytracingShader() 
	{
	
	}
}
