#include "Window.h"

int main(int argc, char** argv) {
	LightRay::Window testWindow("test", 1000, 800);
	
	while (testWindow.IsRunning()) {
		testWindow.Update();
	}	
	return 0;
}
