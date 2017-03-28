#include "CollisionDetectionEvents.h"
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
		Options::showObjectIntersections = !Options::showObjectIntersections;
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

static void O(bool down) {
	if (down) {
		cout << "Toggling Octree!" << endl;
		Options::showOctree = !Options::showOctree;
		Update();
		Refresh();
	}
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

void LeftMouse(bool down, int mods) {
	md.leftDown = down;
}

void RightMouse(bool down, int mods) {
}

void Events::key_cb(GLFWwindow* window, int key, int scancode, int action, int mods) {
	bool down = action == GLFW_PRESS || action == GLFW_REPEAT;
	switch (key) {
	case GLFW_KEY_R: R(down);
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

void Events::Initialize() {
	Data::quadtree->build(Data::polygons);
	Update();
}

void Events::Update() {
	using namespace Data;
	Sketcher::instance()->clear();
	if (Options::showOctree)
		Sketcher::instance()->add(*quadtree);

	if (Options::showObjects || Options::showQuantizedObjects)
		Sketcher::instance()->add(*polygons);
}


void Events::Refresh() {
	using namespace GLUtilities;

	/* Clear stuff */
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for (int i = 1; i < 11; ++i) {

		glm::mat4 matrix(1.0);
		matrix = glm::rotate(matrix, i * .001f, glm::vec3{ 0.0f, 0.0f, 1.0f });
		Data::polygons->movePolygon(i-1, matrix);
	}
	/* Move objects */
	Data::quadtree->build(Data::polygons);
	Update();
	/* Draw objects */
	if (Options::showInstructions)
		Sketcher::instance()->drawPlanes(OrthoCamera::MV);

	if (Options::showOctree)
		Sketcher::instance()->drawBoxes(OrthoCamera::MV);

	if (Options::showObjects || Options::showQuantizedObjects)
		Sketcher::instance()->drawLines(OrthoCamera::MV);

	if (Options::showObjectVertices || Options::showResolutionPoints)
		Sketcher::instance()->drawPoints(OrthoCamera::MV);

	glfwSwapBuffers(GLUtilities::window);
}

