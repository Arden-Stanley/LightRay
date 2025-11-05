#pragma once
#include "Window.h"
#include "Common.h"
//#include <glm/glm.hpp>
#include "Camera.h"

namespace LR 
{
	class EventSystem 
	{
		public:
			EventSystem(Window* window, Camera* camera);
			~EventSystem();
			
			void processInput(float deltaTime);

			bool IsKeyPressed(int key) const;
			bool IsMouseButtonPressed(int button) const;
			void GetMousePosition(double& x, double& y) const;

			float moveSpeed; //TODO: Move to camera class
		private:
			Window* m_window;
			Camera* m_camera;
	};
}