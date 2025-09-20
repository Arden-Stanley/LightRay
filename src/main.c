#include "window.h"
#include "shader.h"

int main(int argc, char** argv) {
	Window* window = create_window(1000, 800, "Test");
	
	Shader screen_shader = create_texture_shader("", "");
	start_window(window);
	
	while (is_window_running(window)) {
		update_window(window);
	}

	end_window(window);

	return 0;
}
