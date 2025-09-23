#include "Window.h"
#include "Shader.h"
#include "Buffer.h"
#include "Common.h"

int main(int argc, char** argv) 
{
	std::unique_ptr<LR::Window> window = std::make_unique<LR::Window>(1000, 800, "Test");	

	LR::RenderShader bufferShader
	(
		"C:\\Users\\Arden Stanley\\source\\repos\\Arden-Stanley\\LightRay\\src\\shaders\\vertex.glsl", 
		"C:\\Users\\Arden Stanley\\source\\repos\\Arden-Stanley\\LightRay\\src\\shaders\\fragment.glsl"
	);

	LR::RaytracingShader rtShader
	(
		"C:\\Users\\Arden Stanley\\source\\repos\\Arden-Stanley\\LightRay\\src\\shaders\\raytracer.glsl"
	);

	LR::Buffer screenBuffer;

	while(window->IsRunning())
	{
		bufferShader.Use();
		screenBuffer.Render();
		window->Update();
	}

	return 0;
}
