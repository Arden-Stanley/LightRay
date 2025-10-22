#ifndef BUFFER_H
#define BUFFER_H

#include "Common.h"
#include "Window.h"
#include "Shader.h"
#include "Raytracing/Renderer.h"

namespace LR 
{
	class Buffer 
	{
		public:
			Buffer(const std::unique_ptr<Window> &window);
			~Buffer();
			void render(const Shader &shader);
		private:
			unsigned int m_vbo;
			unsigned int m_vao;
			int m_width;
			int m_height;
			unsigned int m_texture;
			Renderer m_renderer;
	};
}

#endif
