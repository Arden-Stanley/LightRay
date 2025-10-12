#include "Window.h"
#include "Shader.h"
#include "Buffer.h"
#include "Common.h"
//ImGui Headers
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>


int main(int argc, char** argv) 
{
	std::unique_ptr<LR::Window> window = std::make_unique<LR::Window>(1000, 1000, "Test");	

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
	ImGuiIO& io = ImGui::GetIO();
	ImGui_ImplGlfw_InitForOpenGL(window->GetGLFWWindow(), true);
	ImGui_ImplOpenGL3_Init();
	ImGui::StyleColorsDark();
	

	static float lightIntensity = 1.0f;
	static bool showDebug = true;
	
	while(window->IsRunning())
	{	
		window->Update();
		
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		//ImGui::ShowDemoWindow();
		ImGui::Begin("Light-Ray Editor");

		//ADD TO SCENE LOGIC HERE
		if (ImGui::Button("Reset Scene")) {
			//scene.Reset();
		}
		if (ImGui::Button("Add Sphere")) {
			//scene.AddSphere();
		}
		//ImGui::SliderFloat("Field of View", &scene.fieldOfView, 10.0f, 120.0f);
		//ImGui::SliderInt("Ray Depth", &scene.rayDepth, 1, 10);
		//ImGui::Checkbox("Show Debug Info", &showDebug);
		ImGui::End();

		screenBuffer->Render(rtShader, bufferShader);

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

	//ImGui Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	return 0;
}
