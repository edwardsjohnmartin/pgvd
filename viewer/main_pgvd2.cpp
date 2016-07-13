#pragma optimize("", off)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <fstream>
#include "../timer.h"
#define GL_LOG_FILE "gl.log"

#include "./gl_utils.h"

#include "clfw.hpp"

#include "gl_utils.h"

#include "../opencl/vec.h"
#include "./Polylines.h"
#include "./opengl/Shaders.hpp"
#include "Line.h"
#include "./LinesProgram.h"
#include "./Octree2.h"
#include "../Resln.h"
#include "z_order.h"
// keep track of window size for things like the viewport and the mouse cursor
int g_gl_width = 512;
int g_gl_height = 512 ;
GLFWwindow* g_window = NULL;
cl_float count1 = 0;
cl_float count2 = 0;

float2 point1, point2;
PolyLines* lines;
LinesProgram* program;
Octree2* octree;

bool leftMouseDown = false;
bool rightMouseDown = false;
float2 curMouse;

GLFWcursor* arrowCursor;
GLFWcursor* zoomCursor;
bool zoomMode = false;

// Draws octree and lines.
void refresh() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  if (options.showOctree)
    octree->render(program);
  
  octree->renderBoundingBox(Shaders::boxProgram);
  lines->render(program);
  glfwSwapBuffers(g_window);
}

// key_callback
void onKey(GLFWwindow* window, int key, int scancode,
           int action, int mods) {
  using namespace std;

  if (action == GLFW_PRESS) {
    switch (key) {
      case GLFW_KEY_C:
        lines->clear();
        octree->build(*lines);
        break;
      case GLFW_KEY_P:
        options.showObjectVertices = !options.showObjectVertices;
        break;
      case GLFW_KEY_O:
        options.showOctree = !options.showOctree;
        break;
      case GLFW_KEY_Z:
        zoomMode = !zoomMode;
        if (zoomMode) {
          glfwSetCursor(g_window, zoomCursor);
        } else {
          glfwSetCursor(g_window, arrowCursor);
        }
        break;
      case GLFW_KEY_Q:
        glfwSetWindowShouldClose(g_window, 1);
        break;
    }
  }

  refresh();
}

void onMouse(GLFWwindow* window, int button, int action, int mods) {
  using namespace std;

  if (action == GLFW_PRESS) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      leftMouseDown = true;
      rightMouseDown = false;
      lines->newLine(curMouse);
      octree->build(*lines);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
      lines->setPoint(curMouse, true);
      rightMouseDown = true;
      leftMouseDown = false;
      octree->build(*lines);
    }
  } else if (action == GLFW_RELEASE) {
      leftMouseDown = false;
      rightMouseDown = false;
  }
}

void onMouseMove(GLFWwindow* window, double xpos, double ypos) {
  using namespace std;

  const float x = (xpos / g_gl_width) * 2 - 1;
  const float y = (ypos / g_gl_height) * 2 - 1;
  curMouse = make_float2(x, -y);

  if (leftMouseDown) {
    lines->addPoint(curMouse);
    octree->build(*lines);
    refresh();
  }
  else if (rightMouseDown) {
    lines->setPoint(curMouse, false);
    octree->build(*lines);
    refresh();
  }
}

void InitializeOpenGL() {
  restart_gl_log();
  start_gl();
  print_error("new a");
  glfwSetWindowTitle(g_window, "Parallel GVD");

  glfwSetKeyCallback(g_window, onKey);
  glfwSetMouseButtonCallback(g_window, onMouse);
  glfwSetCursorPosCallback(g_window, onMouseMove);

  glfwSetCursor(g_window, arrowCursor);

  GLFWcursor* arrowCursor = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
  GLFWcursor* zoomCursor = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);


  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  glfwSwapBuffers(g_window);
}

void checkError(cl_int error) {
  if (error != CL_SUCCESS) {
    cout << "ERROR initializing OpenCL!" << endl;
    getchar();
    std::exit(-1);
  }
}

int main(int argc, char** argv) {
  using namespace std;

  checkError(CLFW::Initialize(false, true));

  InitializeOpenGL();
  Shaders::create();

  octree = new Octree2();
  octree->processArgs(argc, argv);
  lines = new PolyLines();
  lines->newLine({ 0.0f, 0.0f });
  lines->addPoint({ 0.0f, 0.0f});
  point1 = point2 = make_float2(0.0f, 0.0f);

  program = new LinesProgram();

  //glm::mat4 matrix = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f);
  //program->setMatrix(matrix);
	
  /*for (const string& f : options.filenames) {
    ifstream in(f.c_str());
    bool first = true;
    while (!in.eof()) {
      double x, y;
      in >> x >> y;
      if (options.jitter) {
        double s = 100;
        x += rand() / (RAND_MAX*s) - 1/(s*2);
        y += rand() / (RAND_MAX*s) - 1/(s*2);
      }
      if (!in.eof()) {
        if (first) {
          lines->newLine(make_float2(x, y));
          first = false;
        } else {
          lines->addPoint(make_float2(x, y));
        }
      }
    }
    in.close();
  }
  rebuild();

  refresh();

*/
  //float radius = 1.0;
  ////500000
  //for (int i = 0; i < 3000; i++) {
  //  radius -= .0005;
  //  addpt(radius);
  //  rebuild();
  //}
  while (!glfwWindowShouldClose(g_window)) {
    glfwPollEvents();
  }
  glfwTerminate();
  return 0;
}
