#ifndef WINDOW_H
#define WINDOW_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
	GLFWwindow* buffer; 
	int width;
	int height;
	char* title;
} Window; 

Window* create_window(int width, int height, char* title);

void start_window(Window* window);

void update_window(Window* window);

bool is_window_running(Window* window);

void end_window(Window* window);

#endif
