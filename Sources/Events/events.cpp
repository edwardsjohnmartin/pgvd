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

// glm::mat4 mvMatrix(1.0);
glm::mat4 mvMatrix(0.2);

static MouseData md;
static GLFWcursor* crossHairCursor;
static floatn point1 = make_floatn(-1.0, -1.0);
static floatn point2 = make_floatn(1.0, 1.0);

static void C(bool down) {
  if (down) {
    cout << "Clearing!" << endl;
    Data::lines->clear();
    //octree->build(*lines);
  }
}

static void P(bool down) {
    if (down) {
      cout << "Toggling Points!" << endl;
      Options::showObjectVertices = !Options::showObjectVertices;
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
    // mvMatrix = glm::mat4(1.0);
    mvMatrix = glm::mat4(0.8);
  }
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
  // md.leftDown = down;
  if ((mods & GLFW_MOD_SHIFT) != 0 && (mods & GLFW_MOD_CONTROL) == 0) {
    Data::lines->addPoint({ md.x, -md.y });
    Data::octree->build(Data::lines);
    refresh();
  } else if ((mods & GLFW_MOD_SHIFT) != 0 && (mods & GLFW_MOD_CONTROL) != 0) {
    glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(-md.x, md.y, 0.0f));
    glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(.5f, .5f, 1.0f));
    mvMatrix = S * T * mvMatrix;
  } else if ((mods & GLFW_MOD_CONTROL) != 0) {
    glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(-md.x, md.y, 0.0f));
    glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 2.0f, 1.0f));
    mvMatrix = S * T * mvMatrix;
  } else {
    Data::lines->newLine({ md.x, -md.y });
  }
}

void RightMouse(bool down, int mods) {
    md.rightDown = down;
    //lines->setPoint(curMouse, true);
    //octree->build(*lines);
}

void key_cb(GLFWwindow* window, int key, int scancode, int action, int mods) {
  bool down = action == GLFW_PRESS || action == GLFW_REPEAT;
  switch (key) {
    case GLFW_KEY_S:
      if (down) Options::showSketcher = !Options::showSketcher;
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
  }
}

void mouse_cb(GLFWwindow* window, int button, int action, int mods) {
  if (action == GLFW_PRESS) {
    button == GLFW_MOUSE_BUTTON_LEFT ? LeftMouse(DOWN, mods) : RightMouse(DOWN, mods);
    md.leftDown = DOWN;
  }
  else if (action == GLFW_RELEASE) {
    // button == GLFW_MOUSE_BUTTON_LEFT ? LeftMouse(UP, mods) : RightMouse(UP, mods);
    md.leftDown = UP;
  }
}

void mouse_move_cb(GLFWwindow* window, double xpos, double ypos) {

    md.x = (xpos / window_width) * 2 - 1;
    md.y = (ypos / window_height) * 2 - 1;

    if (md.leftDown) {
        Sketcher::instance()->clear();

        Data::lines->addPoint({ md.x, -md.y });
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
  assert(glGetError() == GL_NO_ERROR);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  assert(glGetError() == GL_NO_ERROR);
  if (Options::showOctree) {
    Data::octree->draw(mvMatrix);
    assert(glGetError() == GL_NO_ERROR);
  }
  if (Options::showObjects) {
    Data::lines->render(mvMatrix);
    assert(glGetError() == GL_NO_ERROR);
  }

  if (Options::showSketcher) {
    using namespace GLUtilities;
    Sketcher::instance()->draw(mvMatrix);
  }

  glfwSwapBuffers(GLUtilities::window);
}
