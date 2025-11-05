#ifndef BUFFER_H
#define BUFFER_H

#include "Common.h"
#include "Window.h"
#include "Shader.h"
#include "Raytracing/Renderer.h"
#include "Camera.h"

namespace LR 
{
	class Buffer 
	{
		public:
			Buffer(const std::unique_ptr<Window> &window);
			~Buffer() = default;
			void render(const Shader &shader, const Camera &camera);
		private:
			unsigned int m_vbo;
			unsigned int m_vao;
			int m_width;
			int m_height;
			unsigned int m_texture;
			std::unique_ptr<Renderer> m_renderer;
	};
}

#endif
