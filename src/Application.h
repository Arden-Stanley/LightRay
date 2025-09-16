#ifndef APPLICATION_H
#define APPLICATION_H

#include "Window.h"
#include "Shader.h"


namespace LightRay {
	class Application {
		public:
			Application();
			~Application();
			void Run() const;

				
		private:
			std::unique_ptr<Window> _window;	
	};
}

#endif
