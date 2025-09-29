#include "Window.h"
#include "Shader.h"
#include "Buffer.h"
#include "Common.h"

//ImGui Headers
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


int main(int argc, char** argv) 
{
	std::unique_ptr<LR::Window> window = std::make_unique<LR::Window>(800, 800, "Test");	

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

	//ImGui Initialization
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window->GetGLFWwindow(), true);
	ImGui_ImplOpenGL3_Init("#version 330");

	static float lightIntensity = 1.0f;
	static bool showDebug = true;

	while(window->IsRunning())
	{
		screenBuffer->Render(rtShader, bufferShader);
		window->Update();

		//ImGui Frame Start
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Light-Ray Editor");
		//ADD TO SCENE LOGIC HERE
		if (ImGui::Button("Add Sphere")) {
			// TODO: Hook this into your scene logic
		}


	}

	return 0;
}
