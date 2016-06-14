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
#include "./LinesProgram.h"
#include "./Octree2.h"
#include "../Resln.h"

// keep track of window size for things like the viewport and the mouse cursor
int g_gl_width = 512;
int g_gl_height = 512 ;
GLFWwindow* g_window = NULL;
cl_float count1 = 0;
cl_float count2 = 0;

Polylines* lines;
LinesProgram* program;
Octree2* octree;

bool mouseDown = false;
float2 curMouse;

GLFWcursor* arrowCursor;
GLFWcursor* zoomCursor;
bool zoomMode = false;

//------------------------------------------------------------
// refresh
//------------------------------------------------------------
void refresh() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  if (options.showOctree) {
    octree->render(program);
  }
  lines->render(program);
  glfwSwapBuffers(g_window);
}

void rebuild() {
  octree->build(*lines);
  refresh();
}

//------------------------------------------------------------
// key_callback
//------------------------------------------------------------
void onKey(GLFWwindow* window, int key, int scancode,
           int action, int mods) {
  using namespace std;

  if (action == GLFW_PRESS) {
    switch (key) {
      case GLFW_KEY_SPACE:
        lines->addPoint({(cl_float).2*sin(count1),(cl_float).2*cos(count2)});
        count1+=.01;
        count2+=.01;
        rebuild();
        break;
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

void addpt(double radius) {
  lines->newLine({ (cl_float)radius*sin(count1),(cl_float)radius*cos(count2) });
  count1 += 1;
  count2 += 1;
  lines->addPoint({ (cl_float)radius*sin(count1),(cl_float)radius*cos(count2) });
}

void onMouse(GLFWwindow* window, int button, int action, int mods) {
  using namespace std;

  if (action == GLFW_PRESS) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      mouseDown = true;
      lines->newLine(curMouse);
      rebuild();
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    }
  } else if (action == GLFW_RELEASE) {
      mouseDown = false;
  }
}

void onMouseMove(GLFWwindow* window, double xpos, double ypos) {
  using namespace std;

  const float x = (xpos / g_gl_width) * 2 - 1;
  const float y = (ypos / g_gl_height) * 2 - 1;
  curMouse = make_float2(x, -y);

  if (mouseDown) {
    lines->addPoint(curMouse);
    rebuild();
  }
}

void pollEvents() {
  while (true) {
  }
}

int main(int argc, char** argv) {
  using namespace std;

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
  // glfwSetCursor(g_window, zoomCursor);

  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glClear (GL_COLOR_BUFFER_BIT);

  cl_int error = CLFW::Initialize(true);
  if (error != CL_SUCCESS) {
    cout << "ERROR initializing OpenCL!" << endl;
    getchar();
    std::exit(-1);
  }

  octree = new Octree2();
  octree->processArgs(argc, argv);
  lines = new Polylines();

  program = new LinesProgram();
	
  for (const string& f : options.filenames) {
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

  //float radius = 1.0;
  ////500000
  //for (int i = 0; i < 3000; i++) {
  //  radius -= .0005;
  //  addpt(radius);
  //}
  while (!glfwWindowShouldClose(g_window)) {
    // Refresh here for animation
    rebuild();
    glfwPollEvents();
  }
  glfwTerminate();
  return 0;
}
