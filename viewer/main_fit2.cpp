#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#define GL_LOG_FILE "gl.log"

#include "gl_utils.h"

#include "../opencl/vec.h"
#include "./Polylines.h"
#include "./LinesProgram.h"
#include "./Octree2.h"
#include "../Resln.h"
#include "../Karras.h"

// keep track of window size for things like the viewport and the mouse cursor
int g_gl_width = 500;
int g_gl_height = 400;
GLFWwindow* g_window = NULL;

Polylines* lines;
LinesProgram* program;
Octree2* octree;

bool mouseDown = false;
float2 curMouse;

GLFWcursor* arrowCursor;
GLFWcursor* zoomCursor;
bool zoomMode = false;

void fit();

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
  // octree->build(*lines);
  fit();
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

bool ctrl;
void onMouse(GLFWwindow* window, int button, int action, int mods) {
  using namespace std;

  if (action == GLFW_PRESS) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      mouseDown = true;
      // lines->newLine(curMouse);
      ctrl = mods & GLFW_MOD_SHIFT;
      if (ctrl) {
        lines->replacePoint(curMouse, 1, 1);
      } else {
        lines->replacePoint(curMouse, 0, 1);
      }
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
    if (ctrl) {
      lines->replacePoint(curMouse, 1, 1);
    } else {
      lines->replacePoint(curMouse, 0, 1);
    }
    // lines->addPoint(curMouse);
    rebuild();
  }
}

const char *byte_to_binary(int x)
{
  static const int size = 16;//sizeof(x) * 8;
  static char b[size+1];
  b[0] = '\0';

  for (int z = 1<<(size-1); z > 0; z >>= 1) {
    strcat(b, (x & z) ? "1" : "0");
  }

  return b;
}

int get_position(const float coord, const float split, const float w) {
  if (coord < split-w) return 0;
  if (coord < split) return 1;
  if (coord < split+w) return 2;
  return 3;
}

void fit() {
  options.showOctree = true;

  Resln resln(1<<options.max_level);


  BoundingBox<float2> bb;
  bb(make_float2(-0.5, -0.5));
  bb(make_float2(0.5, 0.5));
  // octree->build(*lines, &bb);
  // octree->build(*lines);
  // octree->build(points, &bb);

  // The first two points are the first line segment, and the second two points
  // are the second line segment.
  vector<float2> fpoints;
  fpoints.push_back(lines->getPolygons()[0][0]);
  fpoints.push_back(lines->getPolygons()[0][1]);
  fpoints.push_back(lines->getPolygons()[1][0]);
  fpoints.push_back(lines->getPolygons()[1][1]);
  vector<intn> points = Karras::Quantize(fpoints, resln, &bb);

  // The two lines a and b in parametric form (p = a_p0 + t*a_v).
  // floatn a_p0 = lines->getPolygons()[0][0];
  // floatn a_v = lines->getPolygons()[0][1] - a_p0;
  // floatn b_p0 = lines->getPolygons()[1][0];
  // floatn b_v = lines->getPolygons()[1][1] - b_p0;
  intn a_p0 = points[0];
  intn a_v = points[1] - a_p0;
  intn b_p0 = points[2];
  intn b_v = points[3] - b_p0;

  // Given two points where p0.x == p1.x, find the split point between
  // the two using binary search.
  //
  //     ___________________
  //    |                   |
  //    |                   |
  //    |                   |
  //    |                   |
  //    |                   |
  //    |                   |
  //    |                   |
  //    |__________x__x_____|
  //
  //     ___________________
  //    |         |         |
  //    |         |         |
  //    |         |         |
  //    |_________|_________|
  //    |         |         |
  //    |         |         |
  //    |         |         |
  //    |_________|x__x_____|
  //
  //     ___________________
  //    |         |         |
  //    |         |         |
  //    |         |         |
  //    |_________|_________|
  //    |         |    |    |
  //    |         |____|____|
  //    |         |    |    |
  //    |_________|x__x|____|
  //
  //    Split in quadrant 01: splits = 01
  //     ___________________
  //    |         |         |
  //    |         |         |
  //    |         |         |
  //    |_________|_________|
  //    |         |    |    |
  //    |         |____|____|
  //    |         |_|__|    |
  //    |_________|x|_x|____|
  //
  //    Split in quadrant 00: splits = 00 01
  // float s = bb.min().x;
  int s = 0;
  // float cur_y = a_p0.y;
  int cur_y = 0;
  // float w = bb.size().x;
  int w = resln.width;
  vector<OctNode> nodes;
  int splits = 0;
  int numSplits = 1;
  const int LEFT = 0;
  const int RIGHT = 1;
  int shift = 0;
  // Initial split
  nodes.push_back(OctNode());
  w >>= 1;
  s = s + w;
  int ax = a_p0.x;
  int bx = b_p0.x;
  if (ax > bx) {
    swap(ax, bx);
  }
  while (s < ax || s > bx) {
    w >>= 1;
    if (s < ax) {
      s = s + w;
      splits |= (RIGHT << shift);
      nodes.back().set_child(1, numSplits);
    } else {
      s = s - w;
      splits |= (LEFT << shift);
      nodes.back().set_child(0, numSplits);
    }
    nodes.push_back(OctNode());
    shift += 2;
    numSplits++;
  }

  cout << " splits = " << byte_to_binary(splits) << endl;

  int a_position = 0;
  int b_position = 0;
  cur_y += w;
  // while (abs(a_position - b_position) < 3 && w < ???) {
  {
    // There's potential for a conflict cell if the a and b positions are
    // not completely separated.
    // Preconditions:
    //    * cur_y is at the point to check in this iteration.
    //    * w is the width of check segments (see below).

    // Check the middle line of the current node and record the position.
    //
    //  0    1 | 2    3
    //     ____|____
    //         |
    //       * | *
    //
    // So at y = split.y + w, 
    //      x value      position
    //    < split.x-w       0
    //    < split.x         1
    //    < split.x+w       2
    //     otherwise        3

    cur_y += w;
    float a_t = (cur_y - a_p0.y) / a_v.y;
    float b_t = (cur_y - b_p0.y) / b_v.y;
    int a_x = (int)((a_p0.x + a_t * a_v.x) + 0.5);
    int b_x = (int)((b_p0.x + b_t * b_v.x) + 0.5);
    a_position = get_position(a_x, s, w);
    b_position = get_position(b_x, s, w);
    cout << "a = " << a_position << " b = " << b_position << endl;

    cur_y += w;
    w <<= 1;
  }


  octree->set(nodes, bb);

  // karras_points.clear();
  // bb = BoundingBox<float2>();
  // extra_qpoints.clear();
  // octree.clear();

  // vector<vector<float2>> temp_polygons = lines.getPolygons();
  // if (temp_polygons.empty()) {
  //   buildOctVertices();
  //   return;
  // }

  // // karras_points.clear();
  // vector<vector<float2> > all_vertices(temp_polygons.size());
  // for (int i = 0; i < temp_polygons.size(); ++i) {
  //   const vector<float2>& polygon = temp_polygons[i];
  //   all_vertices[i] = polygon;
  //   for (int j = 0; j < polygon.size()-1; ++j) {
  //     karras_points.push_back(temp_polygons[i][j]);
  //   }
  //   karras_points.push_back(temp_polygons[i].back());
  // }

  // // Find bounding box of vertices
  // for (int j = 0; j < all_vertices.size(); ++j) {
  //   const std::vector<float2>& vertices = all_vertices[j];
  //   for (int i = 0; i < vertices.size(); ++i) {
  //     bb(vertices[i]);
  //   }
  // }
  // if (customBB) {
  //   bb = *customBB;
  // }

  // // Karras iterations
  // vector<intn> qpoints = Karras::Quantize(karras_points, resln);
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

  GLFWcursor* arrowCursor = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
  GLFWcursor* zoomCursor = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
  glfwSetCursor(g_window, arrowCursor);
  // glfwSetCursor(g_window, zoomCursor);

  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glClear (GL_COLOR_BUFFER_BIT);

  octree = new Octree2();
  octree->processArgs(argc, argv);
  lines = new Polylines();

  lines->newLine(make_float2(-0.2, -0.5));
  lines->addPoint(make_float2(-0.24, 0.5));
  lines->newLine(make_float2(-0.18, -0.5));
  lines->addPoint(make_float2(-0.01, 0.5));
  fit();

  program = new LinesProgram();
	
  refresh();

  while (!glfwWindowShouldClose(g_window)) {
    // Refresh here for animation
    // refresh();
    
    glfwPollEvents ();
  }
	
  glfwTerminate();
  return 0;
}
