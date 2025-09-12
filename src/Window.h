/*
 * File Contents:
 * 	-LightRay::Window
 * File Description:
 * 	-This file contains the Window class under the namespace LightRay which acts as a wrapper class
 *  	 around GLFW windows.  As such, the LightRay::Window abstraction can be used to more
 *  	 effectively work with data in an encapsulated and easier to deal with way.
 */

#ifndef WINDOW_H
#define WINDOW_H

#include "Common.h"

namespace LightRay {

	/*
	 * Description:
	 * 	-A class which acts as a 'wrapper' around the GLFW procedural implementation
	 * 	 of a system agnostic window.
	 * 	-Allows for ease of utilizing and modifying window functionality within 
	 * 	 the scope of LightRay.
	 */
	class Window {
		public:
			/*
			 * Description:
			 * 	-Handle GLFW and OpenGL (GLAD) initialization,
			 * 	 creates a GLFW window with specified attributes
			 * 	 and stores it in _buffer.
			 * Parameters:
			 * 	-const std::string &title:
			 * 		-The desired title (as a string or literal) that
			 * 		 is to be displayed on window.	 
			 *	-int width:
			 *		-The desired width in pixels of the window.
			 *	-int height:
			 *		-The desired height in pixels of the window.
			 */ 
			Window(const std::string &title, int width, int height);
		
			/*
			 * Description:
			 * 	-Do some shutdown work, terminate GLFW services and destroy GLFW window object (_buffer).
			 */	
			~Window();
		
			/*
			 * Description:
			 * 	-Performs frame-by-frame updates to the window by clearing the screen,
			 * 	 swapping buffers, and polling for window events.
			 */	
			void Update() const;
			
			/*
			 * Description:
			 * 	-Allows for user to determine if the window
			 * 	 is still active and running.
			 * Return:
			 * 	-A boolean in which true indicates that the window
			 * 	 is still in fact running, and false indicates that 
			 * 	 the window has had a close event triggered.
			 */
			bool IsRunning() const;
		private:
			/*
			 * Description:
			 * 	-A pointer to the underlying GLFW implementation of a window, which
			 * 	 this LightRay::Window class is wrapped around.
			 */
			GLFWwindow *_buffer;

			/*
			 * Description:
			 * 	-A string which keeps track of the title that is displayed on the window.
			 */
			std::string _title;

			/*
			 * Description:
			 * 	-The width and the height in pixels of the window.
			 */
			int _width, _height;
	};
}

#endif
