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
			void use() const;
			void setUniform1i(const std::string &name, int value) const;
		private:
			typedef enum {
				VERTEX,
				FRAGMENT
			} Type;
			unsigned int m_program;
			unsigned int m_loadShader(const std::string &path, Type shaderType) const;

	};
}

#endif
