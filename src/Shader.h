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
		protected:
			typedef enum 
			{
				VERTEX,
				FRAGMENT,
				COMPUTE	
			} Type; 
			unsigned int m_LoadShader(const std::string &path, Type shaderType) const;
	};

	class RenderShader : protected Shader
	{
		public:
			RenderShader(const std::string &vertexPath, const std::string &fragmentPath);
			~RenderShader();
		private:
			unsigned int m_program;
	};

	class RaytracingShader : protected Shader
	{
		public:
			RaytracingShader(const std::string &computePath);
			~RaytracingShader();
		private:
			unsigned int m_program;
	};
}

#endif
