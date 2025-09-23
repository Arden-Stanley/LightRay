#ifndef BUFFER_H
#define BUFFER_H

#include "Common.h"
#include "Window.h"

namespace LR 
{
	class Buffer 
	{
		public:
			Buffer(const Window &window);
			~Buffer();
			void Render() const;
		private:
			unsigned int m_vbo;
			unsigned int m_vao;
			int m_width;
			int m_height;
			unsigned int m_texture;
	};
}

#endif
