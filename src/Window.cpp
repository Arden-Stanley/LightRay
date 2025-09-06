#include "Window.h"

namespace LR {
	Window::Window(const std::string &title, int width, int height) 
	: _buffer(nullptr), _title(title), _width(width), _height(height) {
		glfwInit();
		_buffer = glfwCreateWindow(_width, _height, _title.c_str(), NULL, NULL);	
	}

	Window::~Window() {
		glfwTerminate();	
	}

	void Window::Update() const {
		glfwSwapBuffers(_buffer);
		glfwPollEvents();
	}

	bool Window::IsRunning() const {
		return !glfwWindowShouldClose(_buffer);
	}
}
