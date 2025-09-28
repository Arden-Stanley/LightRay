#ifndef SHADER_H
#define SHADER_H

#include "Common.h"

namespace LR 
{
	class Shader 
	{
		public:
			~Shader() = default;
			void Use() const;
			void SetUniform1i(const std::string &name, int value) const;
			void SetUniform1f(const std::string &name, float value) const;
		protected:
			unsigned int m_program;
			typedef enum 
			{
				VERTEX,
				FRAGMENT,
				COMPUTE	
			} Type; 
			unsigned int m_LoadShader(const std::string &path, Type shaderType) const;
	};

	class RenderShader : public Shader
	{
		public:
			RenderShader(const std::string &vertexPath, const std::string &fragmentPath);
			~RenderShader();
	};

	class RaytracingShader : public Shader
	{
		public:
			RaytracingShader(const std::string &computePath);
			~RaytracingShader();
	};
}

#endif
