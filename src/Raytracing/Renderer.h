#ifndef RENDERER_CUH
#define RENDERER_CUH

#include <iostream>

namespace LR {
    class Renderer {
        public:
            Renderer() = default;
            Renderer(unsigned int texHandle, int screenWidth, int screenHeight);
            ~Renderer();
            void render();
        private:
            unsigned int m_texHandle;
            int m_screenWidth, m_screenHeight;
    };
}

#endif