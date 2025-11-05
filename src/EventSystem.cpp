#include "EventSystem.h"
#include <GLFW/glfw3.h>
#include <iostream>

namespace LR 
{
	EventSystem::EventSystem(Window* window, Camera* camera) 
	: m_window(window), m_camera(camera) {}

	EventSystem::~EventSystem() {}

	void EventSystem::processInput(float deltaTime)
	{
		// Exit program with the ESC key
		if (IsKeyPressed(GLFW_KEY_ESCAPE)) 
		{
			glfwSetWindowShouldClose(m_window->getGLFWWindow(), true);
		}
		moveSpeed = 5.0f * deltaTime;
		//Camera controls
		if (IsKeyPressed(GLFW_KEY_W))
			m_camera->Move(Position{0, 0, -moveSpeed});
		if (IsKeyPressed(GLFW_KEY_S))
			m_camera->Move(Position{0, 0, moveSpeed});
		if (IsKeyPressed(GLFW_KEY_A))
			m_camera->Move(Position{-moveSpeed, 0, 0});
		if (IsKeyPressed(GLFW_KEY_D))
			m_camera->Move(Position{moveSpeed, 0, 0});
	}
	bool EventSystem::IsKeyPressed(int key) const 
	{
		return m_window->isKeyPressed(key);
	}
	bool EventSystem::IsMouseButtonPressed(int button) const 
	{
		return glfwGetMouseButton(m_window->getGLFWWindow(), button) == GLFW_PRESS;
	}
	void EventSystem::GetMousePosition(double& x, double& y) const 
	{
		glfwGetCursorPos(m_window->getGLFWWindow(), &x, &y);
	}
}