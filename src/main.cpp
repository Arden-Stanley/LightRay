#include "Window.h"
#include "Shader.h"
#include "Buffer.h"
#include "Common.h"

int main(int argc, char** argv) 
{
	std::unique_ptr<LR::Window> window = std::make_unique<LR::Window>(1000, 562, "Test");	

	const std::string SOURCE_DIRECTORY = std::string(SOURCE_DIR);
	LR::RenderShader bufferShader
	(
		SOURCE_DIRECTORY + "/src/shaders/vertex.glsl",
		SOURCE_DIRECTORY + "/src/shaders/fragment.glsl"
	);

	LR::RaytracingShader rtShader
	(
	 	SOURCE_DIRECTORY + "/src/shaders/raytracer.glsl"
	);

	std::unique_ptr<LR::Buffer> screenBuffer = std::make_unique<LR::Buffer>(window);

	while(window->IsRunning())
	{
		screenBuffer->Render(rtShader, bufferShader);
		window->Update();
	}

	return 0;
}
