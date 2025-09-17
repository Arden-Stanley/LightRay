#include "window.h"

Window* create_window(int width, int height, char* title) {
	if (!glfwInit()) {
		fputs("ERROR: Could not initialized GLFW properly.\n", stderr);
	}
	Window* window = (Window *)malloc(sizeof(Window));	
	window->width = width;
	window->height = height;
	window->title = title;
	window->buffer = glfwCreateWindow(width, height, title, NULL, NULL);
	if (!window->buffer) {
		fputs("ERROR: Could not create window with GLFW.\n", stderr);	
	}
	return window;
}

void start_window(Window* window) {
	glfwMakeContextCurrent(window->buffer);	
	if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
		fputs("ERROR: Could not load GL functions with GLAD.\n", stderr);
	};
}

void update_window(Window* window) {
	glClear(GL_COLOR_BUFFER_BIT);
	glfwSwapBuffers(window->buffer);
	glfwPollEvents();
}

bool is_window_running(Window* window) {
	return !glfwWindowShouldClose(window->buffer);
}

void end_window(Window* window) {
	glfwDestroyWindow(window->buffer);
	free(window);
	glfwTerminate();
}
