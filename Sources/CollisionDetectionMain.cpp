#include <iostream>
#include "GLUtilities/gl_utils.h"
#include "clfw.hpp"
#include "Events/CollisionDetectionEvents.h"
#include "Shaders/Shaders.hpp"
#include "Polylines/Polylines.h"
#include "GLUtilities/Polygons.h"
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
		glfwSetWindowTitle(window, "2D Collision Detection Demo");
		InitializeGLFWEventCallbacks();
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POLYGON_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
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


vector<vector<float2>> objects;
void generateGear(int numTeeth, float toothThickness, float scale) {
	Gear gear;
	gear.numTeeth = 2 * numTeeth;
	gear.toothThickness = toothThickness;
	gear.outerRadius = gear.numTeeth * gear.toothThickness / M_PI;
	gear.innerRadius = gear.outerRadius - .1;
	gear.dAngle = ((gear.toothThickness * 2) / (gear.numTeeth * 2 * gear.toothThickness)) * 2 * M_PI;

	for (int i = 0; i < numTeeth * 2; i += 2) {
		float theta1 = i * gear.dAngle - (gear.dAngle / 2);
		float theta2 = (i + .15) * gear.dAngle - (gear.dAngle / 2);
		float theta3 = (i + .85) * gear.dAngle - (gear.dAngle / 2);
		float theta4 = (i + 1) * gear.dAngle - (gear.dAngle / 2);
		float2 p1 = make_float2(sin(theta1) * gear.innerRadius, cos(theta1) * gear.innerRadius) * scale;
		float2 p2 = make_float2(sin(theta2) * gear.outerRadius, cos(theta2) * gear.outerRadius) * scale;
		float2 p3 = make_float2(sin(theta3) * gear.outerRadius, cos(theta3) * gear.outerRadius) * scale;
		float2 p4 = make_float2(sin(theta4) * gear.innerRadius, cos(theta4) * gear.innerRadius) * scale;
		gear.points.push_back(p1);
		gear.points.push_back(p2);
		gear.points.push_back(p3);
		gear.points.push_back(p4);
	}
	gear.outerRadius *= scale;
	gear.innerRadius *= scale;
	Data::gears.push_back(gear);
	objects.push_back(gear.points);
}

int main(int argc, char** argv) {
	using namespace std;
	using namespace Data;
	processArgs(argc, argv);

	CLFW::Initialize(false, Options::computeDevice, Options::cl_options, 2);
	GLFW::Initialize();

	Shaders::create();

	/*Data::gearInfo.R = 128;
	Data::gearInfo.S = 64;
	Data::gearInfo.P = 32;*/

	//generateGear(Data::gearInfo.R, .1, .099);
	//generateGear(Data::gearInfo.S, .1, .099);
	//generateGear(Data::gearInfo.P, .1, .1);

	generateGear(20, .1, .1);
	generateGear(20, .1, .1);
	generateGear(20, .1, .1);
	generateGear(20, .1, .1);
	generateGear(20, .1, .1);
	generateGear(20, .1, .1);
	generateGear(20, .1, .1);
	generateGear(20, .1, .1);
	generateGear(20, .1, .1);
	generateGear(20, .1, .1);
	generateGear(20, .1, .1);
	generateGear(20, .1, .1);
	generateGear(20, .1, .1);

	Data::polygons = new Polygons(objects);

	/*for (int i = 0; i < gears.size(); ++i) {
		gears[i].matrix = glm::rotate(gears[i].matrix, gears[i].dAngle / 2, glm::vec3(0, 0, 1.0));
		if (i % 2) {
			gears[i].matrix = glm::rotate(gears[i].matrix, gears[i].dAngle, glm::vec3(0, 0, 1.0));
		}
		Data::polygons->movePolygon(i, gears[i].matrix);
	}*/
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
