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
			void Update() const;
			bool IsRunning() const;
			int GetWidth() const;
			int GetHeight() const;
			GLFWwindow* GetGLFWWindow() const;
			bool IsKeyPressed(int key) const;
		private:
			int m_width, m_height;
			std::string m_title;
			GLFWwindow* m_window;			
	};
}

#endif
