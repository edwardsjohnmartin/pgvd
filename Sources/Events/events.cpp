/*
* C - clears all lines.
* P - Show points
* O - Show octree
* Z - Enter zoom mode
* Q - Quit
*/

#include "events.h"
#include "mouse.h"
#include "GLUtilities/OrthoCamera.h"
#include <iostream>
using namespace std;
using namespace GLUtilities;

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
	}
}

static void two(bool down) {
	if (down) {
		cout << "Toggling Points" << endl;
		Options::showObjectVertices = !Options::showObjectVertices;
	}
}

static void C(bool down) {
	if (down) {
		cout << "Clearing!" << endl;
		Data::lines->clear();
		//octree->build(*lines);
	}
}

static void P(bool down) {
		if (down) {
			cout << "Toggling pruning" << endl;
			Options::pruneOctree = !Options::pruneOctree;
			Data::quadtree->build(Data::lines);
		}
}

static void O(bool down) {
		if (down) {
			cout << "Toggling Octree!" << endl;
			Options::showOctree = !Options::showOctree;
		}
}

static void Z(bool down) {
}

static void R(bool down) {
	if (down) {
		OrthoCamera::reset();
	}

	Data::quadtree->build(Data::lines);


		// mvMatrix = glm::scale(mvMatrix, glm::vec3(2.0f, 2.0f, 2.0f));
		// cout << "Toggling Zoom Mode!" << endl;
		// Options::zoomMode = down;
		// if (Options::zoomMode) {
		//     if (!crossHairCursor)
		//         crossHairCursor = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
		//     md.cursor = crossHairCursor;
		//     glfwSetCursor(window, md.cursor);
		// }
		// else {
		//     glfwSetCursor(window, NULL);
		// }
}

static void Q(bool down) {
	if (down) {
		cout << "Quitting!" << endl;
		glfwSetWindowShouldClose(GLUtilities::window, 1);
	}
}

void LeftMouse(bool down, int mods) {
	md.leftDown = down;
	glm::vec4 temp = glm::vec4(md.x, -md.y, 0.0, 1.0);
	temp = OrthoCamera::IMV * temp;

	/* If we're holding down shift but not control */
	if ((mods & GLFW_MOD_SHIFT) != 0 && (mods & GLFW_MOD_CONTROL) == 0) {
		Data::lines->addPoint({ temp.x, temp.y });
		Data::quadtree->build(Data::lines);
		refresh();
	} /* Else if we're pressing shift but not control */
	else if ((mods & GLFW_MOD_SHIFT) != 0 && (mods & GLFW_MOD_CONTROL) == 0) {
		OrthoCamera::zoom(glm::vec2(temp.x, temp.y), .75);
	} /* else we're only pressing control */
	else if (mods & GLFW_MOD_CONTROL) {
		OrthoCamera::zoom(glm::vec2(-temp.x, -temp.y), 1.0/.75);
	} /* else we're not pressing any modifiers */
	else {
		Data::lines->newLine({ temp.x, temp.y });
	}
}

void RightMouse(bool down, int mods) {
	//OrthoCamera::pan({ md.x, md.y });
		//lines->setPoint(curMouse, true);
		//octree->build(*lines);
}

void key_cb(GLFWwindow* window, int key, int scancode, int action, int mods) {
	bool down = action == GLFW_PRESS || action == GLFW_REPEAT;
	switch (key) {
		case GLFW_KEY_S:
			//if (down) Options::showSketcher = !Options::showSketcher;
			break;
		case GLFW_KEY_C: C(down);
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
		case GLFW_KEY_1: one(down);
			break;
		case GLFW_KEY_2: two(down);
			break;
	}
}

void mouse_cb(GLFWwindow* window, int button, int action, int mods) {
	if (action == GLFW_PRESS) {
		button == GLFW_MOUSE_BUTTON_LEFT ? LeftMouse(DOWN, mods) : RightMouse(DOWN, mods);
		button == GLFW_MOUSE_BUTTON_LEFT ? md.leftDown = DOWN : md.rightDown = DOWN;
	}
	else if (action == GLFW_RELEASE) {
		button == GLFW_MOUSE_BUTTON_LEFT ? LeftMouse(UP, mods) : RightMouse(UP, mods);
		button == GLFW_MOUSE_BUTTON_LEFT ? md.leftDown = UP : md.rightDown = UP;
	}
}

void mouse_move_cb(GLFWwindow* window, double xpos, double ypos) {
	float oldx = md.x; float oldy = md.y;
	md.x = (xpos / window_width) * 2 - 1;
	md.y = (ypos / window_height) * 2 - 1;

	if (md.leftDown) {
		Sketcher::instance()->clear();
		glm::vec4 temp = glm::vec4(md.x, -md.y, 0.0, 1.0);
		temp = OrthoCamera::IMV * temp;

		Data::lines->addPoint({ temp.x, temp.y });
		Data::quadtree->build(Data::lines);
		refresh();
	}

	if (md.rightDown) {
		glm::vec4 newVec = { md.x, md.y, 0.0, 1.0 };
		glm::vec4 oldVec = { oldx, oldy, 0.0, 1.0 };
		newVec = OrthoCamera::IMV  * newVec;
		oldVec = OrthoCamera::IMV  * oldVec;
		OrthoCamera::pan({ newVec.x - oldVec.x, -(newVec.y - oldVec.y) });
	}
}

void scroll_cb(GLFWwindow* window, double xoffset, double yoffset) {
	int sign = std::signbit(yoffset);
	glm::vec4 temp = glm::vec4(md.x, -md.y, 0.0, 1.0);
	temp = OrthoCamera::IMV * temp;
	if (sign == 1)
		OrthoCamera::zoom(glm::vec2(temp.x, temp.y), .90);
	else
		OrthoCamera::zoom(glm::vec2(temp.x, temp.y), 1.0/.90);
}

void resize_cb(GLFWwindow* window, int width, int height) {
		GLUtilities::window_width = width;
		GLUtilities::window_height = height;
		glViewport(0, 0, width, height);
}

void focus_cb(GLFWwindow* window, int focused) {

}

void refresh() {
	using namespace GLUtilities;

	/* Clear stuff */
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	Sketcher::instance()->clear();
  
	/* Add stuff to draw */
	Sketcher::instance()->add(*Data::quadtree);
	Sketcher::instance()->add(*Data::lines);

	/* Draw it */
	Sketcher::instance()->draw(OrthoCamera::MV);

	glfwSwapBuffers(GLUtilities::window);

}
