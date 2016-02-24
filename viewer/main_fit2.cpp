#include <fstream>
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

const int NUM_CELLS = (1 << DIM);

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
        ofstream cache("cache.dat");
        cache << lines->getPolygons()[0][0] << endl;
        cache << lines->getPolygons()[0][1] << endl;
        cache << lines->getPolygons()[1][0] << endl;
        cache << lines->getPolygons()[1][1] << endl;
        cache.close();

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
// Stack sizes are currently limited to 32 elements. This
// should be tied to the max depth of the octree.
//------------------------------------------------------------
const int STACK_SIZE = 32;

struct IntStack {
  int values[STACK_SIZE];
  // Points to the top of the stack
  int position;
};
void push(IntStack* stack, const int value) {
  stack->values[++stack->position] = value;
}
int pop(IntStack* stack) {
  return stack->values[stack->position--];
}
int top(IntStack* stack) {
  return stack->values[stack->position];
}

struct OctNodes {
  OctNode* array;
  int count;
};
void addOctNode(OctNodes* nodes, const OctNode node) {
  if (nodes->array) {
    nodes->array[nodes->count++] = node;
  } else {
    nodes->count++;
  }
}
void setChild(OctNodes* nodes, const int parentIndex, const int position) {
  if (nodes->array) {
    nodes->array[parentIndex].set_child(position, nodes->count);
  }
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
  int w;
  // Represents our current location in the quadtree. See above for examples.
  int positionStack;

  //--------------------------------------------------
  // The following variables relate to construction
  // of the octree. The nodes data member 
  //--------------------------------------------------
  // The octree. Requires an a priori count of how many
  // splits we'll be making.
  OctNodes* nodes;
  // This is a stack, similar to positionStack, which keeps track of
  // indices into the nodes vector (see above).
  IntStack indexStack;
};

WalkState createWalkState(const int w, OctNodes* nodes) {
  WalkState state;
  state.origin = make_intn(0, 0);
  state.w = w;
  state.positionStack = 0;
  // state.indexStack = {
  //   .position = 0
  // };
  state.indexStack.position = 0;
  state.nodes = nodes;
  return state;
}

void split(WalkState* state, const int position) {
  if (position > -1) {
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
    const int parentIndex = top(&state->indexStack);
    setChild(state->nodes, parentIndex, position);

    state->w >>= 1;
  }

  push(&state->indexStack, state->nodes->count);
  addOctNode(state->nodes, OctNode());
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
int getRegion(const float coord, const float split, const float w) {
  if (coord < split-w) return 0;
  if (coord < split) return 1;
  if (coord < split+w) return 2;
  return 3;
}

void popLevel(WalkState* state) {
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
          make_intn(state->origin[0] - state->w, state->origin[1] - state->w);
      break;
  }
  
  // Back out one level
  state->w <<= 1;
  // state->indexStack.pop_back();
  pop(&state->indexStack);
  (state->positionStack) >>= 2;
}

int getInterceptRegion(const WalkState* state,
    const int y_value, const int y_axis, const intn p0, const floatn v) {
  const int x_axis = 1 - y_axis;
  const float t = (y_value - p0[y_axis]) / v[y_axis];
  const int x =  p0[x_axis] + int(v[x_axis] * t);
  const int x_center = state->origin[x_axis] + (state->w >> 1);
  const int region =
      getRegion(x, x_center, (state->w>>1));
  return region;
}

struct CellIntercepts {
  // Three y intercepts: origin, center, and top. One x intercept: center.
  int ybottom;
  int ycenter;
  int ytop;
  int xcenter;
};

// Intuit as walking along the y axis. However, it works if walking
// along the x axis as well. If wxaxis=0 then we're walking along the
// x, otherwise y.
void getIntercepts(
    const WalkState* state,
    const int waxis, const intn p0, const floatn v,
    CellIntercepts* intercepts) {
  
  const int x_axis = 1-waxis;
  const int y_axis = waxis;

  intercepts->ybottom = getInterceptRegion(
      state, state->origin[y_axis], y_axis, p0, v);
  intercepts->ycenter = getInterceptRegion(
      state, state->origin[y_axis] + (state->w >> 1), y_axis, p0, v);
  intercepts->ytop = getInterceptRegion(
      state, state->origin[y_axis] + state->w, y_axis, p0, v);
  intercepts->xcenter = getInterceptRegion(
      state, state->origin[x_axis] + (state->w >> 1), x_axis, p0, v);
}

// Contains subdivide tasks for between 0 and 2^DIM subdivisions
struct ToSubdivide {
  // The number of subdivisions to do. Must be no greater than 2^DIM.
  int total;
  // The number of subdivisions completed.
  int i;
  // The subdivision quadrants.
  int quadrants[1<<DIM];
};

void addSubdivision(ToSubdivide* ret, const int quadrant) {
  const int i = ret->total;
  ret->quadrants[i] = quadrant;
  ret->total++;
}

int getIntersectedQuadrants(const CellIntercepts* intercepts) {
  int ret = 0;
  // ybottom
  if (intercepts->ybottom == 1) {
    ret |= (1 << 0);
  } else if (intercepts->ybottom == 2) {
    ret |= (1 << 1);
  }
  // ycenter
  if (intercepts->ycenter == 1) {
    ret |= (1 << 0);
    ret |= (1 << 2);
  } else if (intercepts->ycenter == 2) {
    ret |= (1 << 1);
    ret |= (1 << 3);
  }
  // ytop
  if (intercepts->ytop == 1) {
    ret |= (1 << 2);
  } else if (intercepts->ytop == 2) {
    ret |= (1 << 3);
  }
  // xcenter
  if (intercepts->xcenter == 1) {
    ret |= (1 << 0);
    ret |= (1 << 1);
  } else if (intercepts->xcenter == 2) {
    ret |= (1 << 2);
    ret |= (1 << 3);
  }

  return ret;
}

ToSubdivide createToSubdivide(
    const WalkState* state, const int waxis,
    CellIntercepts* a, CellIntercepts* b) {
  ToSubdivide ret;
  ret.total = 0;
  ret.i = 0;
  const int a_quads = getIntersectedQuadrants(a);
  const int b_quads = getIntersectedQuadrants(b);
  int conflicts = a_quads & b_quads;

  // If waxis is the x axis, then we need to swap quadrants 1 and 2.
  if (waxis == 0) {
    const int temp = conflicts;
    conflicts &= ~(1<<1);
    conflicts &= ~(1<<2);
    if (temp & (1<<1)) {
      conflicts |= (1<<2);
    }
    if (temp & (1<<2)) {
      conflicts |= (1<<1);
    }
  }

  for (int i = 0; i < NUM_CELLS; ++i) {
    if (conflicts & (1 << i)) {
      addSubdivision(&ret, i);
    }
  }
  return ret;
}

ToSubdivide createToSubdivide(
    const WalkState* state, const int waxis,
    const intn a_p0, const floatn a_v,
    const intn b_p0, const floatn b_v) {
  CellIntercepts a_intercepts, b_intercepts;
  getIntercepts(state, waxis, a_p0, a_v, &a_intercepts);
  getIntercepts(state, waxis, b_p0, b_v, &b_intercepts);
  return createToSubdivide(
      state, waxis, &a_intercepts, &b_intercepts);
}

void fit(intn a_p0, floatn a_v,
                    intn b_p0, floatn b_v,
             int w_, OctNodes* nodes) {
  // Resln resln(1<<options.max_level);
  Resln resln = make_resln(1<<options.max_level);
  WalkState state = createWalkState(w_, nodes);

  // waxis is the walk axis
  const int waxis = (a_v.x < a_v.y) ? 1 : 0;
  const int axis = 1-waxis;

  // Do the initial split.
  split(&state, -1);

  vector<ToSubdivide> toSubdivideStack;

  ToSubdivide toSubdivide =
      createToSubdivide(&state, waxis, a_p0, a_v, b_p0, b_v);
  toSubdivideStack.push_back(toSubdivide);

  while (!toSubdivideStack.empty()) {
    toSubdivide = toSubdivideStack.back();
    toSubdivideStack.pop_back();

    if (toSubdivide.i < toSubdivide.total) {
      // There's a subdivision at this level to do
      const int i = toSubdivide.i++;
      toSubdivideStack.push_back(toSubdivide);
      const int quadrant = toSubdivide.quadrants[i];
      split(&state, quadrant);

      toSubdivide =
          createToSubdivide(&state, waxis, a_p0, a_v, b_p0, b_v);
      toSubdivideStack.push_back(toSubdivide);
    } else {
      // Back out one state
      popLevel(&state);
    }
  }
}

void fit() {
  options.showOctree = true;

  // Resln resln(1<<options.max_level);
  Resln resln = make_resln(1<<options.max_level);

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
  // if (length(fpoints[3]-fpoints[1]) < length(fpoints[2]-fpoints[0])) {
  //   swap(fpoints[0], fpoints[1]);
  //   swap(fpoints[2], fpoints[3]);
  // }
  if (fpoints[0][1] > fpoints[1][1]) {
    swap(fpoints[0], fpoints[1]);
    swap(fpoints[2], fpoints[3]);
  }

  using namespace OctreeUtils;

  vector<intn> points = Karras::Quantize(fpoints, resln, &bb, true);

  intn a_p0 = points[0];
  floatn a_v = convert_floatn(points[1] - a_p0);
  intn b_p0 = points[2];
  floatn b_v = convert_floatn(points[3] - b_p0);

  int w = resln.width;

  // First call. Get the number of octree nodes.
  OctNodes nodes;
  nodes.array = 0;
  nodes.count = 0;
  fit(a_p0, a_v, b_p0, b_v, w, &nodes);

  // Second call. Populate the nodes array.
  nodes.array = new OctNode[nodes.count];
  nodes.count = 0;
  fit(a_p0, a_v, b_p0, b_v, w, &nodes);

  // Put in a vector and create the octree object.
  vector<OctNode> vnodes(nodes.array, nodes.array+nodes.count);
  delete [] nodes.array;
  octree->set(vnodes, bb);
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

  ifstream cache("cache.dat");
  if (cache) {
    floatn p;
    cache >> p;
    lines->newLine(p);
    cache >> p;
    lines->addPoint(p);
    cache >> p;
    lines->newLine(p);
    cache >> p;
    lines->addPoint(p);
  } else {
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
