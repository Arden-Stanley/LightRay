#include "Window.h"

namespace LR {
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
		glClear(GL_COLOR_BUFFER_BIT);
		glfwSwapBuffers(m_window);
		glfwPollEvents();
	}

	bool Window::IsRunning() const 
	{
		return !glfwWindowShouldClose(m_window);
	}
}
