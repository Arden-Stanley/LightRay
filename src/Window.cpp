/*
 * File Description:
 * 	-Implementation file of the "Window.h" file.
 * 	-Implements the member functions for the LightRay::Window class.
 * 	-For general member function documentation, check "Window.h".
 */

#include "Window.h"

namespace LightRay {
	Window::Window(const std::string &title, int width, int height) 
	: _buffer(nullptr), _title(title), _width(width), _height(height) { 
		
		/*
		 * Initialize GLFW systems with correct bindings for system.
		 */	
		glfwInit();

		/*
		 * Create a GLFW window implementation with the parameters,
		 * Store window in _buffer member variable.
		 */
		_buffer = glfwCreateWindow(_width, _height, _title.c_str(), NULL, NULL);
							
		/*
		 * Make the GLFW window we just created (_buffer) the system 
		 * context, load the OpenGL bindings with the system context.
		 */		
		glfwMakeContextCurrent(_buffer);
		gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
	}

	Window::~Window() {
		/*
		 * Destroy the underlying GLFW window pointer (_buffer),
		 * terminate all GLFW processes.
		 */
		glfwDestroyWindow(_buffer);
		glfwTerminate();	
	}

	void Window::Update() const {
		/*
		 * Clears the screen to a solid color, aka the
		 * specified background color of the window.
		 */
		glClear(GL_COLOR_BUFFER_BIT);

		/*
		 * Swaps the back buffer that has been drawn to
		 * with the front buffer which has already been
		 * processed and cleared.
		 */
		glfwSwapBuffers(_buffer);

		/*
		 * Polls all system input events within the 
		 * context of the window.
		 */
		glfwPollEvents();
	}

	bool Window::IsRunning() const {
		/*
		 * Checks if glfwPollEvents() has found a 
		 * glfwWindowShouldClose() event with regards
		 * to the buffer.  
		 * If so, returns false.
		 * If there is none, then this returns true.
		 */
		return !glfwWindowShouldClose(_buffer);
	}
}
