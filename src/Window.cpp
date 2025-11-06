#include "Window.h"
#include <GLFW/glfw3.h>

namespace LR 
{
	Window::Window(int width, int height, const std::string &title) 
		: m_width(width), m_height(height), m_title(title), m_window(nullptr)
    {
		glfwInit();	
		m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), NULL, NULL);
		glfwMakeContextCurrent(m_window);
		gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);	
	}

	Window::~Window() 
	{
		glfwDestroyWindow(m_window);
		glfwTerminate();
	}

	void Window::Update() const 
	{
		glfwSwapBuffers(m_window);
		glfwPollEvents();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	bool Window::IsRunning() const 
	{
		return !glfwWindowShouldClose(m_window);
	}

	int Window::Width() const 
	{
		return m_width;
	}

	int Window::Height() const
	{
		return m_height;
	}

	GLFWwindow* Window::GLFWWindow() const 
	{
		return m_window;
	}
	
	bool Window::IsKeyPressed(int key) const 
	{
		return glfwGetKey(m_window, key) == GLFW_PRESS;
	}
}
