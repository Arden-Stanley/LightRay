/*
 * Description:
 * 	-At the minute using this main function to test out different parts of the raytracer.
 */

#include "Window.h"

int main(int argc, char** argv) {
	
	//Creates a simple window.
	LightRay::Window testWindow("test", 1000, 800);
	
	//Application main loop (happens every render).	
	while (testWindow.IsRunning()) {
		
		testWindow.Update();

	}	
	return 0;
}
