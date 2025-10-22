#ifndef WINDOW_H
#define WINDOW_H

#include "Common.h"

namespace LR 
{
	class Window 
	{
		public:
			Window(int width, int height, const std::string &title);
			~Window();
			void update() const;
			bool isRunning() const;
			int getWidth() const;
			int getHeight() const;
			GLFWwindow* getGLFWWindow() const;
			bool isKeyPressed(int key) const;
		private:
			int m_width, m_height;
			std::string m_title;
			GLFWwindow* m_window;			
	};
}

#endif
