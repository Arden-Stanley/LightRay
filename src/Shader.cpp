#include "Shader.h"

namespace LightRay {
	TextureShader::TextureShader(const std::string &vertexShaderPath, const std::string &fragmentShaderPath) {
		std::ifstream vShaderFile;
		vshaderFile.open(vShaderPath);	
		
		std::string vSrcCode = "";	
		std::string vSrcLine = "";
		while (getline(vShaderFile, vSrcLine)) {
			vSrcCode += vSrcLine;
		}
		const char* vRawSrcCode = vSrcCode.c_str();		
		vShaderFile.close();
		

		unsigned int vShader = glCreateShader(GL_VERTEX_SHADER);

		glShaderSource(vShader, 1, &vRawSrcCode, NULL);
		glCompileShader(vShader);

		int vCompilationStatus;
		glGetShaderiv(vShader, GL_COMPILE_STATUS, &vCompilationStatus);
		if (!vCompilationStatus) {
			int logLength = 1024;
			char message[logLength];

			glGetShaderInfoLog(vShader, logLength, NULL, message);
			std::cout << message << std::endl;
		}
		

		std::ifstream fShaderFile;
		fshaderFile.open(fShaderPath);	
		
		std::string fSrcCode = "";	
		std::string fSrcLine = "";
		while (getline(fShaderFile, srcLine)) {
			fSrcCode += fSrcLine;
		}
		const char* fRawSrcCode = fSrcCode.c_str();		
		fShaderFile.close();
		

		unsigned int fShader = glCreateShader(GL_FRAGMENT_SHADER);

		glShaderSource(fShader, 1, &fRawSrcCode, NULL);
		glCompileShader(fShader);

		int fCompilationStatus;
		glGetShaderiv(fShader, GL_COMPILE_STATUS, &fCompilationStatus);
		if (!fCompilationStatus) {
			int logLength = 1024;
			char message[logLength];

			glGetShaderInfoLog(fShader, logLength, NULL, message);
			std::cout << message << std::endl;
		}

		_shaderID = glCreateProgram();
		glAttachShader(_shaderID, vShader);
		glAttachShader(_shaderID, fShader);
		glLinkProgram(_shaderID);			
	}

	TextureShader::~TextureShader() {}

	void TextureShader::Use() const {
		glUseProgram(_shaderID);	
	}
}
