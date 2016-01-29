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

int changeLine;
int changePoint;
void onMouse(GLFWwindow* window, int button, int action, int mods) {
  using namespace std;

  if (action == GLFW_PRESS) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      mouseDown = true;
      // lines->newLine(curMouse);
      changeLine = (mods & GLFW_MOD_SHIFT) ? 1 : 0;
      changePoint = (mods & GLFW_MOD_CONTROL) ? 0 : 1;
      lines->replacePoint(curMouse, changeLine, changePoint);
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
    lines->replacePoint(curMouse, changeLine, changePoint);
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

int find_split(int s, const int ax, const int bx, int& w, int& parent_quad,
               vector<OctNode>& nodes, vector<int>& nodeIndices,
               const int LEFT, const int RIGHT) {
  while (s < ax || s > bx) {
    w >>= 1;
    const int nodeIndex = nodeIndices.back();
    if (s < ax) {
      s = s + w;
      parent_quad <<= 2;
      parent_quad |= RIGHT;
      nodes[nodeIndex].set_child(RIGHT, nodes.size());
    } else {
      s = s - w;
      parent_quad <<= 2;
      parent_quad |= LEFT;
      nodes[nodeIndex].set_child(LEFT, nodes.size());
    }
    nodeIndices.push_back(nodes.size());
    nodes.push_back(OctNode());
  }
  return s;
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

  using namespace OctreeUtils;

  vector<intn> points = Karras::Quantize(fpoints, resln, &bb);

  // vector<intn> points;
  // vector<CellIntersection> ints = FindIntersections(
  //     convert_floatn(qpoints[0]), convert_floatn(qpoints[1]),
  //     make_intn(0, 0), resln.width, resln);
  // // cout << ints.size() << endl;
  // // cout << ints[0].p << " " << ints[1].p << endl;
  // points.push_back(convert_intn(ints[0].p));
  // points.push_back(convert_intn(ints[1].p));
  // ints = FindIntersections(
  //     convert_floatn(qpoints[1]), convert_floatn(qpoints[2]),
  //     make_intn(0, 0), resln.width, resln);
  // points.push_back(convert_intn(ints[0].p));
  // points.push_back(convert_intn(ints[1].p));

  // The two lines a and b in parametric form (p = a_p0 + t*a_v).
  // floatn a_p0 = lines->getPolygons()[0][0];
  // floatn a_v = lines->getPolygons()[0][1] - a_p0;
  // floatn b_p0 = lines->getPolygons()[1][0];
  // floatn b_v = lines->getPolygons()[1][1] - b_p0;
  intn a_p0 = points[0];
  floatn a_v = convert_floatn(points[1] - a_p0);
  intn b_p0 = points[2];
  floatn b_v = convert_floatn(points[3] - b_p0);

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
  //    parent_quad = xx
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
  //    push split in quadrant 01: parent_quad = 01
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
  //    push split in quadrant 00: parent_quad = 01 00
  int w = resln.width;
  vector<OctNode> nodes;
  // This is a stack, similar to parent_quad, which keeps track of
  // indices into the nodes vector (see above).
  vector<int> nodeIndex;
  // See above for examples.
  int parent_quad = 0;
  const int LOWER_LEFT  = 0;
  const int LOWER_RIGHT = 1;
  const int UPPER_LEFT  = 2;
  const int UPPER_RIGHT = 3;
  // Initial split
  nodeIndex.push_back(nodes.size());
  nodes.push_back(OctNode());
  w >>= 1;
  int s = w;
  int ax = a_p0.x;
  int bx = b_p0.x;
  if (ax > bx) {
    swap(ax, bx);
  }

  s = find_split(s, ax, bx, w, parent_quad, nodes, nodeIndex,
                 LOWER_LEFT, LOWER_RIGHT);

  cout << " parent_quad = " << byte_to_binary(parent_quad) << endl;

  int a_position = 0;
  int b_position = 0;
  intn center = make_intn(s, w);
  vector<int> a_positions, b_positions;
  while (abs(a_position - b_position) < 3 && w < resln.width) {
    // There's potential for a conflict cell if the a and b positions are
    // not completely separated.
    // Preconditions:
    //    * center is the center point of the "plus" to check (see below)
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

    float a_t = (center.y - a_p0.y) / a_v.y;
    float b_t = (center.y - b_p0.y) / b_v.y;
    int a_x = (int)((a_p0.x + a_t * a_v.x) + 0.5);
    int b_x = (int)((b_p0.x + b_t * b_v.x) + 0.5);
    a_position = get_position(a_x, center.x, w);
    b_position = get_position(b_x, center.x, w);
    a_positions.push_back(a_position);
    b_positions.push_back(b_position);

    cout << endl;
    cout << "center = " << center << endl;
    cout << "a_x = " << a_x << " (" << (a_p0.x + a_t * a_v.x) << ") "
         << " b_x = " << b_x << endl;

    const int subdivide_pos = (parent_quad & 3);
    // Recurse and find new split
    if (a_position == b_position) {
      s = find_split(center.x - (w>>1), a_x, b_x, w,
                     parent_quad, nodes, nodeIndex, LEFT, RIGHT);
      center = make_intn(s, center.y+w);
    } else {
    // {
      // Update center, w and parent_quad for next iteration
      if (subdivide_pos == LEFT) {
        center = make_intn(center.x + w, center.y + w);
      } else if (subdivide_pos == RIGHT) {
        center = make_intn(center.x - w, center.y + w);
      } else {
        throw logic_error("Unexpected subdivision position");
      }

      w <<= 1;
      parent_quad >>= 2;
      nodeIndex.pop_back();
    }
  }

  cout << endl;
  cout << "a = ";
  for (int pos : a_positions) {
    cout << pos << " ";
  }
  cout << endl;
  cout << "b = ";
  for (int pos : b_positions) {
    cout << pos << " ";
  }
  cout << endl;

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
  lines->addPoint(make_float2(-0.24, 0.65));
  lines->newLine(make_float2(-0.18, -0.5));
  lines->addPoint(make_float2(-0.01, 0.65));
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
