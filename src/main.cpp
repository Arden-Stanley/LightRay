#include "Window.h"
#include "Shader.h"
#include "Common.h"

int main(int argc, char** argv) 
{
	std::unique_ptr<LR::Window> window = std::make_unique<LR::Window>(1000, 800, "Test");	

	while(window->IsRunning())
	{
		window->Update();
	}
	return 0;
}
