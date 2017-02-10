#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "clfw.hpp"
#include "GLUtilities/gl_utils.h"
#include "GLUtilities/Sketcher.h"
#include "Shaders/Shaders.hpp"
#include <cstdlib>
#include "Kernels/Kernels.h"
#include "Catch/HelperFunctions.hpp"

extern "C" {
#include "BinaryRadixTree/BuildBRT.h"
}

void fixResolution(int& width, int& height) {
	const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
	width = min(width, mode->width);
	height = min(height, mode->height);
	if (width < height) {
		height = width;
	}
	else {
		width = height;
	}
}

void InitializeGLFW(int width = 1000, int height = 1000) {
	using namespace GLUtilities;
	GLUtilities::window_height = height;
	GLUtilities::window_width = width;
	restart_gl_log();
	start_gl();
	fixResolution(width, height);
	GLUtilities::window_height = height;
	GLUtilities::window_width = width;

	print_gl_error();
	glfwSetWindowTitle(window, "Parallel GVD");
	//InitializeGLFWEventCallbacks();
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glfwSwapBuffers(window);
	glfwSwapInterval(1);
}

int main( int argc, char* const argv[] )
{
  /* global setup... */
  CLFW::Initialize(true, true, 2);
	//InitializeGLFW();
	//Shaders::create();
	writeToFile("", "BenchmarkData//binaries//log.txt");

  int result = Catch::Session().run( argc, argv );
  
	//glfwTerminate();
  /* global clean-up... */
  system("pause");
  return result;
}
