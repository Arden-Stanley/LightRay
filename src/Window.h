#ifndef WINDOW_H
#define WINDOW_H

#include <string>
#include <GLFW/glfw3.h>

namespace LR {
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
			 * Performs window updates for each frame.
			 */	
			void Update() const;

			bool IsRunning() const;
		private:
			GLFWwindow *_buffer;
			std::string _title;
			int _width, _height;
	};
}

#endif
