#ifndef SHADER_H
#define SHADER_H

#include "Common.h"

namespace LR 
{
	class Shader
	{
		public:
			Shader(const std::string &vertexPath, const std::string &fragmentPath);
			~Shader();
			void Use() const;
		private:
			typedef enum {
				VERTEX,
				FRAGMENT
			} Type;
			unsigned int m_program;
			unsigned int m_LoadShader(const std::string &path, Type shaderType) const;

	};
}

#endif
