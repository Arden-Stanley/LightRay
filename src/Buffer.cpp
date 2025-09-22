#include "Buffer.h"

namespace LR 
{
	Buffer::Buffer()
		: m_vbo(0), m_vao(0)
	{
		float quad[] =
		{
			-1.0f,  1.0f,
			-1.0f, -1.0f,
			 1.0f, -1.0f,

			 1.0f, -1.0f,
			 1.0f,  1.0f,
			-1.0f,  1.0f
		};

		glGenVertexArrays(1, &m_vao);
		glBindVertexArray(m_vao);

		glGenBuffers(1, &m_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
	}

	Buffer::~Buffer() 
	{
		
	}

	void Buffer::Render() const
	{
		glBindVertexArray(m_vao);
		glDrawArrays(GL_TRIANGLES, 0, 6);
	}
}