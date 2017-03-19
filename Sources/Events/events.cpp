/*
* C - clears all lines.
* P - Show points
* O - Show octree
* Z - Enter zoom mode
* Q - Quit
*/
#include "events.h"
#include "mouse.h"

using namespace GLUtilities;
using namespace Events;

#define DOWN true
#define UP false

static MouseData md;
static GLFWcursor* crossHairCursor;
static floatn point1 = make_floatn(-1.0, -1.0);
static floatn point2 = make_floatn(1.0, 1.0);

static void one(bool down) {
	if (down) {
		cout << "Toggling conflicts" << endl;
		Options::showOctreeConflicts = !Options::showOctreeConflicts;
		Data::quadtree->build(Data::lines);
		Update();
		Refresh();
	}
}

static void two(bool down) {
	if (down) {
		cout << "Toggling Points" << endl;
		Options::showObjectVertices = !Options::showObjectVertices;
		Update();
		Refresh();
	}
}

static void three(bool down) {
	if (down) {
		cout << "Toggling lines" << endl;
		if (Options::showObjects) {
			Options::showObjects = !Options::showObjects;
			Options::showQuantizedObjects = !Options::showQuantizedObjects;
		}
		else if (Options::showQuantizedObjects) {
			Options::showQuantizedObjects = !Options::showQuantizedObjects;
		}
		else if (!Options::showObjects) {
			Options::showObjects = !Options::showObjects;
		}
		Update();
		Refresh();
	}
}

static void C(bool down) {
	if (down) {
		cout << "Clearing!" << endl;
		Data::lines->clear();

		Update();
		Refresh();
	}
}

static void P(bool down) {
	if (down) {
		cout << "Toggling pruning" << endl;
		Options::pruneOctree = !Options::pruneOctree;
		Data::quadtree->build(Data::lines);
		Update();
		Refresh();
	}
}

static void O(bool down) {
	if (down) {
		cout << "Toggling Octree!" << endl;
		Options::showOctree = !Options::showOctree;
		Update();
		Refresh();
	}
}

static void Z(bool down) {
}

static void R(bool down) {
	if (down) {
		OrthoCamera::reset();
	}

	Data::quadtree->build(Data::lines);
	Update();
	Refresh();
}

static void Q(bool down) {
	if (down) {
		cout << "Quitting!" << endl;
		glfwSetWindowShouldClose(GLUtilities::window, 1);
	}
}

static void I(bool down) {
	if (down) {
		cout << "Toggling Image" << endl;
		Options::showImage = !Options::showImage;
	}
}

static void U(bool down) {
	if (down) {
		cout << "Undoing line" << endl;
		Data::lines->undoLine();
		Data::quadtree->build(Data::lines);
	}
	Update();
	Refresh();
}

static void S(bool down) {
	if (down) {
		cout << "Saving lines to folder" << endl;
		Data::lines->writeToFile("./TestData/TestData/vascular_bundles");
	}
}
void LeftMouse(bool down, int mods) {
	md.leftDown = down;
	if (md.leftDown) {
		glm::vec4 temp = glm::vec4(md.x, -md.y, 0.0, 1.0);
		temp = OrthoCamera::IMV * temp;

		/* If we're holding down shift but not control */
		if ((mods & GLFW_MOD_SHIFT) != 0 && (mods & GLFW_MOD_CONTROL) == 0) {
			Data::lines->addPoint({ temp.x, temp.y });
			Data::quadtree->build(Data::lines);
		} /* Else if we're pressing shift but not control */
		else if ((mods & GLFW_MOD_SHIFT) != 0 && (mods & GLFW_MOD_CONTROL) == 0) {
			OrthoCamera::zoom(glm::vec2(temp.x, temp.y), .75);
		} /* else we're only pressing control */
		else if (mods & GLFW_MOD_CONTROL) {
			OrthoCamera::zoom(glm::vec2(-temp.x, -temp.y), 1.0 / .75);
		} /* else we're not pressing any modifiers */
		else {
			Data::lines->newLine({ temp.x, temp.y });
		}

		Update();
		Refresh();
	}
}

void RightMouse(bool down, int mods) {
	//OrthoCamera::pan({ md.x, md.y });
		//lines->setPoint(curMouse, true);
		//octree->build(*lines);
}

void Events::key_cb(GLFWwindow* window, int key, int scancode, int action, int mods) {
	bool down = action == GLFW_PRESS || action == GLFW_REPEAT;
	switch (key) {
	case GLFW_KEY_S: S(down);
		break;
	case GLFW_KEY_C: C(down);
		break;
	case GLFW_KEY_I: I(down);
		break;
	case GLFW_KEY_P: P(down);
		break;
	case GLFW_KEY_O: O(down);
		break;
	case GLFW_KEY_Z: Z(down);
		break;
	case GLFW_KEY_R: R(down);
		break;
	case GLFW_KEY_Q: Q(down);
		break;
	case GLFW_KEY_U: U(down);
		break;
	case GLFW_KEY_1: one(down);
		break;
	case GLFW_KEY_2: two(down);
		break;
	case GLFW_KEY_3: three(down);
		break;
	}
}

void Events::mouse_cb(GLFWwindow* window, int button, int action, int mods) {
	if (action == GLFW_PRESS) {
		button == GLFW_MOUSE_BUTTON_LEFT ? LeftMouse(DOWN, mods) : RightMouse(DOWN, mods);
		button == GLFW_MOUSE_BUTTON_LEFT ? md.leftDown = DOWN : md.rightDown = DOWN;
	}
	else if (action == GLFW_RELEASE) {
		button == GLFW_MOUSE_BUTTON_LEFT ? LeftMouse(UP, mods) : RightMouse(UP, mods);
		button == GLFW_MOUSE_BUTTON_LEFT ? md.leftDown = UP : md.rightDown = UP;
	}
}

void Events::mouse_move_cb(GLFWwindow* window, double xpos, double ypos) {
	float oldx = md.x; float oldy = md.y;
	md.x = (xpos / window_width) * 2 - 1;
	md.y = (ypos / window_height) * 2 - 1;

	if (md.leftDown) {
		Sketcher::instance()->clear();
		glm::vec4 temp = glm::vec4(md.x, -md.y, 0.0, 1.0);
		temp = OrthoCamera::IMV * temp;

		Data::lines->addPoint({ temp.x, temp.y });
		Data::quadtree->build(Data::lines);

		Update();
		Refresh();
	}

	if (md.rightDown) {
		glm::vec4 newVec = { md.x, md.y, 0.0, 1.0 };
		glm::vec4 oldVec = { oldx, oldy, 0.0, 1.0 };
		newVec = OrthoCamera::IMV  * newVec;
		oldVec = OrthoCamera::IMV  * oldVec;
		OrthoCamera::pan({ newVec.x - oldVec.x, -(newVec.y - oldVec.y) });
	}
}

void Events::scroll_cb(GLFWwindow* window, double xoffset, double yoffset) {
	int sign = signbit(yoffset);
	glm::vec4 temp = glm::vec4(md.x, -md.y, 0.0, 1.0);
	temp = OrthoCamera::IMV * temp;
	if (sign == 1)
		OrthoCamera::zoom(glm::vec2(temp.x, temp.y), .90);
	else
		OrthoCamera::zoom(glm::vec2(temp.x, temp.y), 1.0 / .90);
}

void Events::resize_cb(GLFWwindow* window, int width, int height) {
	GLUtilities::window_width = width;
	GLUtilities::window_height = height;
	glViewport(0, 0, width, height);
}

void Events::focus_cb(GLFWwindow* window, int focused) {

}

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
void Events::Initialize() {
	//using namespace GLUtilities;
	//string file = "./TestData/images/Vascular_bundles.png";
	//Sketcher::instance()->uploadImage(file, "test");
	////Sketcher::instance()->uploadObj("./TestData/images/cube.obj", "cube");
	//Plane p = {};
	//p.width = .5;
	//p.height = .5;
	//p.offset = glm::vec3(0.0);
	//p.texName = "test";
	//Sketcher::instance()->add(p);

	//using namespace cv;
	//using namespace std;

	//Mat src, src_gray, canny;
	//vector<Vec4i> hierarchy;
	//int thresh = 100;
	///* Load the source image */
	//src = imread(file, 1);
	///* convert it to gray scale and blur. */
	//cvtColor(src, src_gray, CV_BGR2GRAY);
	//blur(src_gray, src_gray, Size(3, 3));
	///* Detect edges using canny */
	//Canny(src_gray, canny, thresh, thresh * 2, 3);
	///* Find contours */
	//vector<vector<cv::Point> > contours;
	//findContours(canny, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS, cv::Point(3, 3));
	///* Add the contours to polylines */
	//for (int i = 0; i < contours.size(); ++i) {
	//	for (int j = 0; j < contours[i].size(); ++j) {
	//		floatn p = { (float)(contours[i][j].x-1) / src.rows, (float)(contours[i][j].y-1) / src.cols };
	//		p *= 2.0;
	//		p = p - 1.005;
	//		if (j == 0)
	//			Data::lines->newLine(p);
	//		else
	//			Data::lines->addPoint(p);
	//	}
	//}
	Data::quadtree->build(Data::lines);
	Update();
}

void Events::Update() {
	using namespace Data;
	Sketcher::instance()->clear();
	if (Options::showOctree)
		Sketcher::instance()->add(*quadtree);

	if (Options::showObjects)
		Sketcher::instance()->add(*lines);

	if (Options::showQuantizedObjects) {
		Sketcher::instance()->add(
			lines->getQuantizedPolygons(quadtree->bb.minimum, 
				quadtree->resln.width, quadtree->bb.maxwidth
		));
	}
}

void Events::Refresh() {
	using namespace GLUtilities;

	/* Clear stuff */
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/* Draw objects */
	if (Options::showImage)
		Sketcher::instance()->drawPlanes(OrthoCamera::MV);

	if (Options::showOctree)
		Sketcher::instance()->drawBoxes(OrthoCamera::MV);

	if (Options::showObjects || Options::showQuantizedObjects)
		Sketcher::instance()->drawLines(OrthoCamera::MV);

	if (Options::showObjectVertices || Options::showResolutionPoints)
		Sketcher::instance()->drawPoints(OrthoCamera::MV);

	glfwSwapBuffers(GLUtilities::window);
}

