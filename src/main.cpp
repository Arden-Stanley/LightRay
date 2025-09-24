#include "Window.h"
#include "Shader.h"
#include "Buffer.h"
#include "Common.h"

int main(int argc, char** argv) 
{
	std::unique_ptr<LR::Window> window = std::make_unique<LR::Window>(1000, 800, "Test");	

	LR::RenderShader bufferShader
	(
		"../src/shaders/vertex.glsl",
		"../src/shaders/fragment.glsl"
	);

	LR::RaytracingShader rtShader
	(
	 	"../src/shaders/raytracer.glsl"
	);

	LR::Buffer screenBuffer(window);

	while(window->IsRunning())
	{
		screenBuffer.Render(rtShader, bufferShader);
		window->Update();
	}

	return 0;
}
