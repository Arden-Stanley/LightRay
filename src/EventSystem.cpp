#include "EventSystem.h"
#include <GLFW/glfw3.h>
#include <iostream>

namespace LR 
{
	EventSystem::EventSystem(Window* window) : window(window), cameraPosition(0.0f, 0.0f, 0.0f) {}
	EventSystem::~EventSystem() {}
	void EventSystem::processInput(float deltaTime)
	{
		// Exit program with the ESC key
		if (IsKeyPressed(GLFW_KEY_ESCAPE)) 
		{
			glfwSetWindowShouldClose(window->GetGLFWWindow(), true);
		}
		moveSpeed = 5.0f * deltaTime;
		// Camera controls
		if (IsKeyPressed(GLFW_KEY_W))
			cameraPosition.z -= moveSpeed;
		if (IsKeyPressed(GLFW_KEY_S))
			cameraPosition.z += moveSpeed;
		if (IsKeyPressed(GLFW_KEY_A))
			cameraPosition.x -= moveSpeed;
		if (IsKeyPressed(GLFW_KEY_D))
			cameraPosition.x += moveSpeed;

	}
	bool EventSystem::IsKeyPressed(int key) const 
	{
		return window->IsKeyPressed(key);
	}
	bool EventSystem::IsMouseButtonPressed(int button) const 
	{
		return glfwGetMouseButton(window->GetGLFWWindow(), button) == GLFW_PRESS;
	}
	void EventSystem::GetMousePosition(double& x, double& y) const 
	{
		glfwGetCursorPos(window->GetGLFWWindow(), &x, &y);
	}
}