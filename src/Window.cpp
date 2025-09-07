#include "Window.h"

namespace LightRay {
	Window::Window(const std::string &title, int width, int height) 
	: _buffer(nullptr), _title(title), _width(width), _height(height) {
		glfwInit();
		_buffer = glfwCreateWindow(_width, _height, _title.c_str(), NULL, NULL);	
		glfwMakeContextCurrent(_buffer);
		gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
	}

	Window::~Window() {
		glfwDestroyWindow(_buffer);
		glfwTerminate();	
	}

	void Window::Update() const {
		glClear(GL_COLOR_BUFFER_BIT);
		glfwSwapBuffers(_buffer);
		glfwPollEvents();
	}

	bool Window::IsRunning() const {
		return !glfwWindowShouldClose(_buffer);
	}
}
