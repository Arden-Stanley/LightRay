#pragma once
#include "Window.h"
#include "Common.h"
#include <glm/glm.hpp>

namespace LR 
{
	class EventSystem 
	{
		public:
			EventSystem(Window* window);
			~EventSystem();
			
			void processInput(float deltaTime);

			bool IsKeyPressed(int key) const;
			bool IsMouseButtonPressed(int button) const;
			void GetMousePosition(double& x, double& y) const;

			float moveSpeed;
			glm::vec3 cameraPosition;
		private:
			Window* window;
	};
}