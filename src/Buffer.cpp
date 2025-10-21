#include "Buffer.h"

namespace LR 
{
	Buffer::Buffer(const std::unique_ptr<Window> &window)
		: m_vbo(0), m_vao(0), m_width(window->GetWidth()), m_height(window->GetHeight()), m_texture(0)
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
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);

		glBindImageTexture(0, m_texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
		
		startTime = std::chrono::steady_clock::now();
	}

	Buffer::~Buffer() 
	{
		
	}

	void Buffer::Render(const RaytracingShader &raytracer, const RenderShader &bufferShader) const
	{
		std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
		std::chrono::duration<float> duration = endTime - startTime;
		float timeFloat = duration.count();
		//startTime = std::chrono::steady_clock::now();
		raytracer.Use();
		raytracer.SetUniform1f("time", timeFloat);
		glDispatchCompute((unsigned int) m_width, (unsigned int) m_height, 1);
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		bufferShader.Use();
		bufferShader.SetUniform1i("tex", 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_texture);
		glBindVertexArray(m_vao);
		glDrawArrays(GL_TRIANGLES, 0, 6);
	}
}
