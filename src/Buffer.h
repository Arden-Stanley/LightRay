#ifndef BUFFER_H
#define BUFFER_H

#include "Common.h"

namespace LR 
{
	class Buffer 
	{
		public:
			Buffer();
			~Buffer();
			void Render() const;
		private:
			unsigned int m_vbo;
			unsigned int m_vao;
	};
}

#endif
