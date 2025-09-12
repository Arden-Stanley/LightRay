/*
 * File Contents:
 * 	-LightRay::Shader (abstract)
 * 	-Lightray::TextureShader
 * 	-LightRay::RaytracingShader
 * File Description:
 * 	-Seeks to provide an abstraction for GLSL shader 
 * 	 loading and use within LightRay.
 */

#ifndef SHADER_H
#define SHADER_H

#include "Common.h"

namespace LightRay {

	/*
	 * Description: 
	 * 	-Abstract class which provides an inheritable
	 * 	 interface for which different types of shaders
	 * 	 can be created.
	 */
	class Shader {
		public:
			/*
			 * Description:
			 * 	-Default destructor for in case
			 * 	 no proper destructor implemented.
			 */
			~Shader() = default;

			/*
			 * Description:
			 * 	-Virtual function to be implemented by class which inherits.
			 * 	-An interface for telling the program to use the specified 
			 * 	 shader program on its next render pass.
			 */
			virtual void Use() const = 0;	
	};

	/*
	 * Description:
	 * 	-A shader class representing a rasterization shader
	 * 	 which is made up of a fragment and vertex shader program.
	 *	-In the context of LightRay, allows for rendering the
	 *	 2D texture output of the raytracer to the window every frame.
	 * Inherits:
	 * 	-From LightRay::Shader class. 	
	 */
	class TextureShader : Public Shader {
		public:
			/*
			 * Description:
			 * 	-Constructor which loads a GLSL vertex and fragment shader into memory
			 * 	-After loading the files, they are compiled/linked with 
			 * 	 OpenGL functions and stored as a single program, _shaderID.
			 * Parameters:
			 * 	-const std::string &vertexShaderPath: 
			 * 		-A file path specifying where a GLSL vertex shader file with a [*.vertex]
			 * 		 extension is located.
			 * 	-const std::string &fragmentShaderPath:
			 * 		-A file path specifying where a GLSL fragment shader file with a [*.fragment]
			 * 		 extension is located.
			 */
			TextureShader(const std::string &vertexShaderPath, const std::string &fragmentShaderPath);

			/*
			 * Description:
			 * 	-Destroys all associated shaders and compiled shader programs,
			 * 	 as well as any excess shader utilities.
			 */
			~TextureShader();

			/*
			 * Description:
			 * 	-Sets the current shader in use to be the program stored in _shaderID.
			 * Overrides:
			 * 	-LightRay::Shader Use() const;
			 */
			void Use() const override;
		private:
			/*
			 * Description:
			 * 	-Contains an unsigned int which specifies where the 
			 * 	 shader program which is created is stored on the GPU.
			 * 	-For sake of clarity, refer to this as the underlying
			 * 	 shader program itself.
			 */
			unsigned int _shaderID;
	};
	
	/*
	 * Description:
	 * 	-A shader class representing a compute shader
	 * 	 which is made up of a fragment and vertex shader program.
	 *	-In the context of LightRay, allows for leveraging compute
	 *	 shaders to implement raytracing algorithms on the GPU.
	 * Inherits:
	 * 	-From LightRay::Shader class. 	
	 */
	class RaytracingShader : Public Shader {
		public:
			/*
			 * Description:
			 * 	-Constructor which loads and processes a GLSL compute shader files.
			 * 	-After processing computer shader file, it is compiled and
			 * 	 stored as a program in _shaderID.
			 * Parameters:
			 * 	-const std::string &computerShaderPath: 
			 * 		-A file path specifying where a GLSL compute shader file with a [*.compute]
			 * 		 extension is located.
			 */
			RaytracingShader(const std::string &computeShaderPath);	
			
			/*
			 * Description:
			 * 	-Destroys all associated shaders and compiled shader programs,
			 * 	 as well as any excess shader utilities.
			 */	
			~RaytracingShader();
			
			/*
			 * Description:
			 * 	-Sets the current shader in use to be the program stored in _shaderID.
			 * Overrides:
			 * 	-LightRay::Shader Use() const;
			 */
			void Use() const override;
		private:
			/*
			 * Description:
			 * 	-Contains an unsigned int which specifies where the 
			 * 	 shader program which is created is stored on the GPU.
			 * 	-For sake of clarity, refer to this as the underlying
			 * 	 shader program itself.
			 */
			unsigned int _shaderID;
	};
}

#endif
