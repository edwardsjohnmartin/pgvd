/*
* C - clears all lines.
* P - Show points
* O - Show octree
* Z - Enter zoom mode
* Q - Quit
*/

#include "events.h"
#include "mouse.h"
#include <iostream>
using namespace std;
using namespace GLUtilities;

#define DOWN true
#define UP false

static MouseData md;
static GLFWcursor* crossHairCursor;
static void C(bool down) {
  cout << "Clearing!" << endl;
  Data::lines->clear();
  //octree->build(*lines);
}

static void P(bool down) {
  cout << "Toggling Points!" << endl;
  Options::showObjectVertices = !Options::showObjectVertices;
}

static void O(bool down) {
  cout << "Toggling Octree!" << endl;
  if (down)
    Options::showOctree = !Options::showOctree;
}

static void Z(bool down) {
  cout << "Toggling Zoom Mode!" << endl;
  Options::zoomMode = down;
  if (Options::zoomMode) {
    if (!crossHairCursor)
      crossHairCursor = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
    md.cursor = crossHairCursor;
    glfwSetCursor(window, md.cursor);
  } else {
    glfwSetCursor(window, NULL);
  }
}

static void Q(bool down) {
  cout << "Quitting!" << endl;
  glfwSetWindowShouldClose(GLUtilities::window, 1);
}

void LeftMouse(bool down) {
  md.leftDown = down;
  Data::lines->newLine({ md.x, -md.y });
}

void RightMouse(bool down) {
  md.rightDown = down;

  //lines->setPoint(curMouse, true);
  //octree->build(*lines);
}

void key_cb(GLFWwindow* window, int key, int scancode, int action, int mods) {
  bool down = action == GLFW_PRESS || action == GLFW_REPEAT;
  switch (key) {
  case GLFW_KEY_C: C(down);
    break;
  case GLFW_KEY_P: P(down);
    break;
  case GLFW_KEY_O: O(down);
    break;
  case GLFW_KEY_Z: Z(down);
    break;
  case GLFW_KEY_Q: Q(down);
    break;
  }
}

void mouse_cb(GLFWwindow* window, int button, int action, int mods) {
  if (action == GLFW_PRESS)
    button == GLFW_MOUSE_BUTTON_LEFT ? LeftMouse(DOWN) : RightMouse(DOWN);
  else if (action == GLFW_RELEASE)
    button == GLFW_MOUSE_BUTTON_LEFT ? LeftMouse(UP) : RightMouse(UP);
}

void mouse_move_cb(GLFWwindow* window, double xpos, double ypos) {
  md.x = (xpos / window_width) * 2 - 1;
  md.y = (ypos / window_height) * 2 - 1;

  if (md.leftDown) {
    Data::lines->addPoint({md.x, -md.y});
    Data::octree->build(Data::lines);
    refresh();
  }
}

void resize_cb(GLFWwindow* window, int width, int height) {
  GLUtilities::window_width = width;
  GLUtilities::window_height = height;
  glViewport(0, 0, width, height);
}

void focus_cb(GLFWwindow* window, int focused) {

}

void refresh() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  if (Options::showOctree)
    Data::octree->draw();
  if (Options::showObjects)
    Data::lines->render();
  glfwSwapBuffers(GLUtilities::window);
}