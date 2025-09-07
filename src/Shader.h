#ifndef SHADER_H
#define SHADER_H

#include "Common.h"

namespace LightRay {
	class Shader {
		public:
			~Shader() = default;
			virtual void Use() const;	
	};

	class TextureShader : Public Shader {
		public:
			TextureShader(const std::string &vertexShaderPath, const std::string &fragmentShaderPath);
			~TextureShader();
			void Use() const override;
		private:
			unsigned int _shaderID;
	};

	class RaytracingShader : Public Shader {
		public:
			RaytracingShader(const std::string &computeShaderPath);	
			~RaytracingShader();
			void Use() const override;
		private:
			unsigned int _shaderID;
	};
}

#endif
