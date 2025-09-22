#include "Window.h"
#include "Shader.h"
#include "Buffer.h"
#include "Common.h"

int main(int argc, char** argv) 
{
	std::unique_ptr<LR::Window> window = std::make_unique<LR::Window>(1000, 800, "Test");	

	LR::RenderShader shader
	(
		"C:\\Users\\Arden Stanley\\source\\repos\\Arden-Stanley\\LightRay\\src\\shaders\\default.vertex", 
		"C:\\Users\\Arden Stanley\\source\\repos\\Arden-Stanley\\LightRay\\src\\shaders\\default.fragment"
	);

	LR::Buffer screenBuffer;

	while(window->IsRunning())
	{
		shader.Use();
		screenBuffer.Render();
		window->Update();
	}

	return 0;
}
