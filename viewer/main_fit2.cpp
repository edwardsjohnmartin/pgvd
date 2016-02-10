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
void fit(const int axis);

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

void updatePoints();

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
      updatePoints();
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
    updatePoints();
    // lines->addPoint(curMouse);
    rebuild();
  }
}

void updatePoints() {
  for (int i = 0; i < 2; ++i) {
    floatn p0 = lines->getPolygons()[i][0];
    floatn v = lines->getPolygons()[i][1] - p0;

    float pt = (-0.5 - p0.y) / v.y;
    floatn p = p0 + v * pt;
    float qt = (0.5 - p0.y) / v.y;
    floatn q = p0 + v * qt;
    
    bool doSwap = (p.x > q.x);
    if (doSwap) {
      swap(p, q);
    }

    if (p.x < -0.5) {
      pt = (-0.5 - p0.x) / v.x;
      p = p0 + v * pt;
    }
    if (q.x > 0.5) {
      qt = (0.5 - p0.x) / v.x;
      q = p0 + v * qt;
    }

    if (doSwap) {
      swap(p, q);
    }
    lines->replacePoint(p, i, 0);
    lines->replacePoint(q, i, 1);
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

//------------------------------------------------------------
// WalkState struct
// Represents a position in the octree.
//     _______________________
//    |           |           |
//    |           |           |
//    |           |           |
//    |___________|___________|
//    |           |     |     |
//    |           |_____|_____|
//    |           *__|__*     |
//    |___________*__|__*_____|
//
//                |_____|
//            cell of interest
//
// For the cell of interest above, the WalkState struct would
// be the following:
//   positionStack = 01 00
//------------------------------------------------------------
struct WalkState {
  intn origin;
  intn center;
  int w;
  // Represents our current location in the quadtree. See above for examples.
  int positionStack;

  //--------------------------------------------------
  // The following variables relate to construction
  // of the octree and require an a priori count
  // of how many splits we'll be making.
  //--------------------------------------------------
  // The octree
  vector<OctNode> nodes;
  // This is a stack, similar to positionStack, which keeps track of
  // indices into the nodes vector (see above).
  vector<int> indexStack;
};

WalkState createWalkState(const int w) {
  WalkState state;
  state.origin = make_intn(0, 0);
  state.center = make_intn(w>>1, w>>1);
  state.w = w;
  state.positionStack = 0;
  return state;
}

// Should get rid of parentIndex
void split(WalkState* state, const int position) {
  if (state->nodes.empty() && position != -1) {
    throw logic_error("If the state is uninitialized then position must be -1");
  }

  if (!state->nodes.empty()) {
    switch (position) {
      case 0:
        // Origin remains the same
        break;
      case 1:
        state->origin[0] += (state->w >> 1);
        break;
      case 2:
        state->origin[1] += (state->w >> 1);
        break;
      case 3:
        state->origin[0] += (state->w >> 1);
        state->origin[1] += (state->w >> 1);
        break;
    }
    state->positionStack <<= 2;
    state->positionStack |= position;
    const int parentIndex = state->indexStack.back();
    state->nodes[parentIndex].set_child(position, state->nodes.size());

    state->w >>= 1;
  }

  state->indexStack.push_back(state->nodes.size());
  state->nodes.push_back(OctNode());

  state->center = make_intn(state->origin[0] + (state->w>>1),
                            state->origin[1] + (state->w>>1));
}

intn getCenter(const WalkState* state) {
  return state->center;
  // intn center = state->origin;
  // const int w2 = state->w>>1;
  // for (int i = 0; i < DIM; ++i) {
  //   center[i] += w2;
  // }
  // return center;
}

int getPosition(const WalkState* state) {
  return state->positionStack & 3;
}

// Check the middle line of the current node and return the region coord
// is in.
//
//  0    1   2    3    <-- region
//         |
//     ____|____
//         |    
//       * | *  
//                
//         |----|
//           w
int get_region(const float coord, const float split, const float w) {
  if (coord < split-w) return 0;
  if (coord < split) return 1;
  if (coord < split+w) return 2;
  return 3;
}

// Divides in half until a split value is found that is between a and b,
// i.e. uses binary search.
//  a     - min point
//  b     - max point
//  left  - quadrant to split if the current split point s is greater than
//          a and b.
//  right - quadrant to split if the current split point s is less than
//          a and b.
int find_split(intn a_, intn b_, WalkState* state) {
  int ax = a_.x;
  int bx = b_.x;
  int ay = a_.y;
  int by = b_.y;
  const int axis = (abs(ax-bx) > abs(ay-by)) ? 0 : 1;
  const int oaxis = 1-axis;
  if (a_[axis] > b_[axis]) {
    swap(a_, b_);
  }
  const int a = a_[axis];
  const int b = b_[axis];

  int left, right;
  if (axis == 0) {
    if (min(a_[oaxis], b_[oaxis]) == state->origin[oaxis]) {
      left = 0;
      right = 1;
    } else {
      left = 2;
      right = 3;
    }
  } else {
    if (min(a_[oaxis], b_[oaxis]) == state->origin[oaxis]) {
      left = 0;
      right = 2;
    } else {
      left = 1;
      right = 3;
    }
  }

  // The current cell is already subdivided once.
  int s = state->origin[axis] + (state->w>>1);
  while ((s < a || s > b) && (state->w>>1) > 1) {
    int position;
    if (s < a) {
      s = s + (state->w>>2);
      position = right;
    } else {
      s = s - (state->w>>2);
      position = left;
    }
    split(state, position);
  }
  return s;
}

// // Divides in half until a split value is found that is between a and b,
// // i.e. uses binary search.
// //  a     - min point
// //  b     - max point
// //  left  - quadrant to split if the current split point s is greater than
// //          a and b.
// //  right - quadrant to split if the current split point s is less than
// //          a and b.
// int find_split(const int axis, const int a, const int b,
//                WalkState* state,
//                const int left, const int right) {
//   int s = state->origin[axis] + (state->w>>1);
//   while ((s < a || s > b) && (state->w>>1) > 1) {
//     int position;
//     if (s < a) {
//       s = s + (state->w>>2);
//       position = right;
//     } else {
//       s = s - (state->w>>2);
//       position = left;
//     }
//     split(state, position);
//   }
//   return s;
// }

// Computes the center of the parent cell given the child's center,
// half the child's width (w) and the child's position in the parent
// cell (position).
// intn parentCenter(const intn& center, const int w, const int position,
//                   const intn dir) {
//   switch (position) {
//     case 0:
//       return make_intn(center.x + dir[0]*w, center.y + dir[1]*w);
//     case 1:
//       return make_intn(center.x - dir[0]*w, center.y + dir[1]*w);
//     case 2:
//       return make_intn(center.x + dir[0]*w, center.y - dir[1]*w);
//     case 3:
//       return make_intn(center.x - dir[0]*w, center.y - dir[1]*w);
//     default:
//       throw logic_error("Unknown position in parentCenter()");
//   }
// }

void popLevel(WalkState* state, const intn dir) {
  const int position = getPosition(state);
  switch (position) {
    case 0:
      break;
    case 1:
      state->origin =
          make_intn(state->origin[0] - state->w, state->origin[1]);
      break;
    case 2:
      state->origin =
          make_intn(state->origin[0], state->origin[1] - state->w);
      break;
    case 3:
      state->origin =
          make_intn(state->origin[0] - state->w,
                    state->origin[1] - state->w);
      break;
  }
  

  // state->center = parentCenter(
  //     state->center, (state->w>>1), getPosition(state), dir);

  // Back out one level
  state->w <<= 1;
  state->indexStack.pop_back();
  (state->positionStack) >>= 2;
  // cout << "old = " << state->origin;
  // state->origin = make_intn(state->center[0] - (state->w>>1),
  //                           state->center[1] - (state->w>>1));
  // cout << " new = " << state->origin << endl;
  state->center = make_intn(state->origin[0] + (state->w>>1),
                            state->origin[1] + (state->w>>1));
}

// non-const for swap
vector<OctNode> fit(intn a_p0, floatn a_v,
                    intn b_p0, floatn b_v,
                    int w_) {
  Resln resln(1<<options.max_level);

  WalkState state = createWalkState(w_);

  int ax = a_p0.x;
  int bx = b_p0.x;
  int ay = a_p0.y;
  int by = b_p0.y;
  int a, b, axis;
  // The split axis is the axis we search for a split point. So if axis
  // is zero, then we search in x for the split point s, and the split
  // plane becomes the axis-aligned plane at x = s.
  if (abs(ax-bx) > abs(ay-by)) {
    axis = 0;
    // a = ax;
    // b = bx;
  } else {
    axis = 1;
    // a = ay;
    // b = by;
  }
  a = a_p0[axis];
  b = b_p0[axis];
  int oaxis = 1-axis;
  // Make sure a < b.
  if (a > b) {
    swap(a, b);
    swap(a_p0, b_p0);
    swap(a_v, b_v);
  }

  intn dir = make_intn(1, 1);
  const bool negativeDir = (a_v[oaxis] < 0);
  dir[oaxis] = negativeDir ? -1 : 1;
  cout << "dir = " << dir << endl;
  const int dirBit = (dir[oaxis] == -1) ? (1<<oaxis) : 0;
  const int popQuadBit = (1<<oaxis);

  // Do the initial split.
  split(&state, -1);
  int s = find_split(a_p0, b_p0, &state);

  cout << endl;
  cout << "Found initial split. a = " << a << " b = " << b
       << " axis = " << axis << " s = " << s << endl;
  cout << "positionStack = " << byte_to_binary(state.positionStack) << endl;

  vector<int> a_regions, b_regions;

  // Walk along the lines. If they ever both cross the same quadtree edge then
  // separate by subdividing. Call the edge a conflict edge. Two lines
  // crossing a conflict edge will then enter a conflict cell.
  while ((state.w>>1) < resln.width) {
    // Find the region a and b cross at.
    const float a_t = (state.center[oaxis] - a_p0[oaxis]) / a_v[oaxis];
    const float b_t = (state.center[oaxis] - b_p0[oaxis]) / b_v[oaxis];
    const intn a_p = a_p0 + convert_intn(a_v * a_t);
    const intn b_p = b_p0 + convert_intn(b_v * b_t);
    a = (int)((a_p0[axis] + a_t * a_v[axis]) + 0.5);
    b = (int)((b_p0[axis] + b_t * b_v[axis]) + 0.5);
    const int a_region = get_region(a, state.center[axis], (state.w>>1));
    const int b_region = get_region(b, state.center[axis], (state.w>>1));
    a_regions.push_back(a_region);
    b_regions.push_back(b_region);

    cout << endl;
    cout << "checking center = " << state.center << endl;
    cout << "a = " << a
         << " b = " << b << endl;

    if (a_region == b_region) {
      // The region is the same -- this is a conflict edge.
      // Subdivide the cell bordered by the conflict edge.
      cout << "conflict at " << a_region << endl;
      if (a_region == 0 || a_region == 3) {
        cout << "Unsupported region" << endl;
        return state.nodes;
      }
      // Which quadrant the cell to subdivide is in.
      int position = (a_region == 1) ? (1<<oaxis) : 3;
      if (negativeDir) {
        position = (a_region == 1) ? 0 : (1<<axis);
      }
      // Do the initial split.
      split(&state, position);
      s = find_split(a_p, b_p, &state);
    } else {
      // int position = (state.positionStack & 3);
      int position = getPosition(&state);
      // Loop until the cell is on the bottom (if axis = x) or cell is
      // on the left (if axis = y).
      while ((negativeDir?~position:position) & (1<<oaxis)) {
      // while (position & (1<<oaxis)) {
        popLevel(&state, dir);
        position = getPosition(&state);
      }

      popLevel(&state, dir);
      position = getPosition(&state);

    }
  }

  cout << endl;
  cout << "a = ";
  for (int pos : a_regions) {
    cout << pos << " ";
  }
  cout << endl;
  cout << "b = ";
  for (int pos : b_regions) {
    cout << pos << " ";
  }
  cout << endl;

  // octree->set(nodes, bb);
  return state.nodes;

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

  // Order the points such that fpoints[0] is closest to fpoints[2]
  if (length(fpoints[3]-fpoints[0]) < length(fpoints[2]-fpoints[0])) {
    swap(fpoints[2], fpoints[3]);
  }
  if (length(fpoints[3]-fpoints[1]) < length(fpoints[2]-fpoints[0])) {
    swap(fpoints[0], fpoints[1]);
    swap(fpoints[2], fpoints[3]);
  }

  using namespace OctreeUtils;

  vector<intn> points = Karras::Quantize(fpoints, resln, &bb, true);

  intn a_p0 = points[0];
  floatn a_v = convert_floatn(points[1] - a_p0);
  intn b_p0 = points[2];
  floatn b_v = convert_floatn(points[3] - b_p0);

  cout << "a_p0 = " << a_p0 << " " << " b_p0 = " << b_p0 << endl;

  int w = resln.width;
  vector<OctNode> nodes = fit(a_p0, a_v, b_p0, b_v, w);
  octree->set(nodes, bb);
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

  if (options.test_axis == 0) {
    if (options.test_num == 2) {
      // fine
      lines->newLine(make_float2(-0.2, -0.5));
      lines->addPoint(make_float2(-0.24, 0.65));
      lines->newLine(make_float2(-0.01, 0.65));
      lines->addPoint(make_float2(-0.18, -0.5));
    } else if (options.test_num == 1) {
      // medium
      lines->newLine(make_float2(-0.1875,      -0.5));
      lines->addPoint(make_float2(-0.1875-0.02, 0.65));
      lines->newLine(make_float2(-0.0625,      -0.5));
      lines->addPoint(make_float2(-0.0625+0.02, 0.65));
    } else if (options.test_num == 0) {
      // coarse
      lines->newLine(make_float2(-0.375,      -0.5));
      lines->addPoint(make_float2(-0.375-0.02, 0.65));
      // lines->addPoint(make_float2(-0.375+0.01, 0.65));
      lines->newLine(make_float2(-0.125,      -0.5));
      lines->addPoint(make_float2(-0.125+0.02, 0.65));
    }
  } else if (options.test_axis == 1) {
    if (options.test_num == 2) {
      // fine y
      lines->newLine(make_float2(-0.5, -0.2));
      lines->addPoint(make_float2(0.65, -0.24));
      lines->newLine(make_float2(0.65, -0.01));
      lines->addPoint(make_float2(-0.5, -0.18));
    } else if (options.test_num == 1) {
      // medium y
      lines->newLine(make_float2(-0.5, -0.1875));
      lines->addPoint(make_float2(0.65, -0.1875-0.02));
      lines->newLine(make_float2(-0.5, -0.0625));
      lines->addPoint(make_float2(0.65, -0.0625+0.02));
    } else if (options.test_num == 0) {
      // coarse y
      lines->newLine(make_float2(-0.5, -0.375));
      lines->addPoint(make_float2(0.65, -0.375-0.02));
      lines->newLine(make_float2(-0.5, -0.125));
      lines->addPoint(make_float2(0.65, -0.125+0.02));
    }
  }

  updatePoints();

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
