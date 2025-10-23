#include "Buffer.h"

namespace LR 
{
	Buffer::Buffer(const std::unique_ptr<Window> &window)
		: m_vbo(0), m_vao(0), m_width(window->getWidth()), m_height(window->getHeight()), m_texture(0), m_renderer(nullptr)
	{
		float quad[] =
		{   //vertices        //texture coords
			-1.0f,  1.0f,		0.0f,  1.0f,
			-1.0f, -1.0f,		0.0f,  0.0f,
			 1.0f, -1.0f,		1.0f,  0.0f,

			 1.0f, -1.0f,		1.0f,  0.0f,
			 1.0f,  1.0f,		1.0f,  1.0f,
			-1.0f,  1.0f,		0.0f,  1.0f
		};

		glGenVertexArrays(1, &m_vao);
		glBindVertexArray(m_vao);
		
		glGenBuffers(1, &m_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glGenTextures(1, &m_texture);
		glBindTexture(GL_TEXTURE_2D, m_texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
		
		m_renderer = std::make_unique<Renderer>(m_texture, m_width, m_height);
	}


	void Buffer::render(const Shader &shader)
	{
		m_renderer->render();
		shader.use();
		glBindTexture(GL_TEXTURE_2D, m_texture);
		glBindVertexArray(m_vao);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
}
