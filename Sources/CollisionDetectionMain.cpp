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
void generateGear(int numTeeth, float toothThickness) {
	numTeeth *= 2;
	float radius = numTeeth * toothThickness / M_PI;
	float dAngle = ((toothThickness * 2) / (numTeeth * 2 * toothThickness )) * 2 * M_PI;

	vector<float2> object;
	float outerRadius = radius;
	float innerRadius = outerRadius - .1;
	for (int i = 0; i < numTeeth * 2; i += 2) {
		float theta1 = i * dAngle;
		float theta2 = (i + .1) * dAngle;
		float theta3 = (i + .9) * dAngle;
		float theta4 = (i + 1) * dAngle;
		float2 p1 = make_float2(sin(theta1) * innerRadius, cos(theta1) * innerRadius);
		float2 p2 = make_float2(sin(theta2) * outerRadius, cos(theta2) * outerRadius);
		float2 p3 = make_float2(sin(theta3) * outerRadius, cos(theta3) * outerRadius);
		float2 p4 = make_float2(sin(theta4) * innerRadius, cos(theta4) * innerRadius);
		object.push_back(p1);
		object.push_back(p2);
		object.push_back(p3);
		object.push_back(p4);
	}
	objects.push_back(object);
}

int main(int argc, char** argv) {
	using namespace std;
	processArgs(argc, argv);

	CLFW::Initialize(false, Options::computeDevice, Options::cl_options, 2);
	GLFW::Initialize();

	Shaders::create();
	//
	//vector<vector<float2>> objects;
	//vector<float2> line1, line2;
	//line1.push_back(make_float2(0.0, -1.0));
	//line1.push_back(make_float2(0.0,1.0));
	//line2.push_back(make_float2(0.0, -0.5));
	//line2.push_back(make_float2(0.0, 0.5));
	//objects.push_back(line1);
	//objects.push_back(line2);

	/*
	int numPts = 32;
	float innerRadius = .09;
	float outerRadius = .1;
	for (int i = 0; i < numPts; i+=2) {
		float theta1 = i * ((2.0 * M_PI) / numPts);
		float theta2 = (i + .1) * ((2.0 * M_PI) / numPts);
		float theta3 = (i + .9) * ((2.0 * M_PI) / numPts);
		float theta4 = (i + 1) * ((2.0 * M_PI) / numPts);
		float2 p1 = make_float2(sin(theta1) * innerRadius, cos(theta1) * innerRadius);
		float2 p2 = make_float2(sin(theta2) * outerRadius, cos(theta2) * outerRadius);
		float2 p3 = make_float2(sin(theta3) * outerRadius, cos(theta3) * outerRadius);
		float2 p4 = make_float2(sin(theta4) * innerRadius, cos(theta4) * innerRadius);
		object.push_back(p1);
		object.push_back(p2);
		object.push_back(p3);
		object.push_back(p4);
	}
	objects.push_back(object);
	objects.push_back(object);
	objects.push_back(object);*/
	//for (int i = 0; i < object.size(); ++i) {
	//	object[i].x += outerRadius + innerRadius;
	//}
	//objects.push_back(object);

	//object.clear();
	//for (int i = 0; i < numPts; ++i) {
	//	float theta = i * ((2.0 * M_PI) / numPts);
	//	float radius = (in) ? .5 : 1.;
	//	object.push_back(make_float2(sin(theta) * radius, cos(theta))  * radius);
	//	radius = (!in) ? .5 : 1.;
	//	object.push_back(make_float2(sin(theta)  * radius, cos(theta))  * radius);
	//	in = !in;
	//}
	//objects.push_back(object);

	for (int i = 2; i < 50; i+= 4) {
		generateGear(i, .1);
	}
	Data::polygons = new Polygons(objects);
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
