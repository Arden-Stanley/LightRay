#ifndef WINDOW_H
#define WINDOW_H

#include "Common.h"

namespace LightRay {
	class Window {
		public:
			/*
			 * Creates a window object with the specified window title, width, and height.
			 */ 
			Window(const std::string &title, int width, int height);
		
			/*
			 * Do some shutdown work, terminate GLFW services and destroy window.
			 */	
			~Window();
		
			/*
			 * Performs frame-by-frame updates to the window by clearing the screen,
			 * swapping buffers, and polling for window events.
			 */	
			void Update() const;
			
			/*
			 * Returns a boolean value in which true means the window is still running and
			 * has not triggered a glfwWindowShouldClose event on polling.
			 */
			bool IsRunning() const;
		private:
			GLFWwindow *_buffer;
			std::string _title;
			int _width, _height;
	};
}

#endif
