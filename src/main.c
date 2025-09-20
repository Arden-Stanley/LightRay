#include "window.h"
#include "shader.h"

int main(int argc, char** argv) {
	Window* window = create_window(1000, 800, "Test");
	
	start_window(window);
	
	Shader screen_shader = create_texture_shader("../src/shaders/default.vertex", "../src/shaders/default.fragment");
	while (is_window_running(window)) {
		use_shader(screen_shader);
		update_window(window);
	}

	end_window(window);

	return 0;
}
