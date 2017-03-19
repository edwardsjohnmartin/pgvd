#include <iostream>
#include "GLUtilities/gl_utils.h"
#include "clfw.hpp"
#include "Events/events.h"
#include "Shaders/Shaders.hpp"
#include "Polylines/Polylines.h"
#include "GlobalData/data.h"
#include "Options/Options.h"

namespace GLFW {
	void InitializeGLFWEventCallbacks() {
		using namespace GLUtilities;
		glfwSetKeyCallback(window, Events::key_cb);
		glfwSetMouseButtonCallback(window, Events::mouse_cb);
		glfwSetCursorPosCallback(window, Events::mouse_move_cb);
		glfwSetWindowSizeCallback(window, Events::resize_cb);
		glfwSetWindowFocusCallback(window, Events::focus_cb);
		glfwSetScrollCallback(window, Events::scroll_cb);
	}

	void InitializeGLFW(int width = 1000, int height = 1000) {
		using namespace GLUtilities;
		GLUtilities::window_height = height;
		GLUtilities::window_width = width;
		restart_gl_log();
		start_gl();
		GLUtilities::window_height = height;
		GLUtilities::window_width = width;

		print_gl_error();
		glfwSetWindowTitle(window, "Parallel GVD");
		InitializeGLFWEventCallbacks();
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glDisable(GL_DEPTH_TEST);
		glfwSwapBuffers(window);
		glfwSwapInterval(1);
	}
	
	void Initialize() {
		InitializeGLFW();
		InitializeGLFWEventCallbacks();
	}
}

int processArgs(int argc, char** argv) {
	using namespace Options;

	int i = 1;
	bool stop = false;
	while (i < argc && !stop) {
		stop = true;
		if (processArg(i, argv)) {
			stop = false;
		}
	}

	for (; i < argc; ++i) {
		string filename(argv[i]);
		filenames.push_back(filename);
	}
	//TODO: fix this.
	return 0;
}

// Tests
// ./QUADTREE -d 1 -l 24 data/maze/poly*.dat
// 
// These two give different results:
//     ./QUADTREE -d 0 -l 24 data/test1-*.dat // works correctly
//     ./QUADTREE -d 1 -l 24 data/test1-*.dat // doesn't resolve conflict cells (although it does find them)
int main(int argc, char** argv) {
	using namespace std;
	processArgs(argc, argv);

	CLFW::Initialize(false, Options::computeDevice, Options::cl_options, 2);
	GLFW::Initialize();

	Shaders::create();
	
	Data::lines = new PolyLines(Options::filenames);
	Data::quadtree = new Quadtree();

	Events::Initialize();

	/* Event loop */
	while (!glfwWindowShouldClose(GLUtilities::window)) {
		glfwPollEvents();
		Events::Refresh();
	}
	glfwTerminate();
	return 0;
}
