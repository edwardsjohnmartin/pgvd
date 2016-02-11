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
bool isSplit(WalkState* state, const int position) {
  const int parentIndex = state->indexStack.back();
  return !state->nodes[parentIndex].is_leaf(position);
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
void find_split(intn a_, intn b_, WalkState* state) {
  int ax = a_.x;
  int bx = b_.x;
  int ay = a_.y;
  int by = b_.y;
  // The split search axis is based on the proximity of the points
  int axis;
  if (abs(ax-bx) > abs(ay-by)) {
    axis = 0;
  } else if (abs(ax-bx) > abs(ay-by)) {
    axis = 1;
  } else if (ay == state->origin.y || ay == state->origin.y + state->w) {
    axis = 0;
  } else {
    axis = 1;
  }
  
  const int oaxis = 1-axis;
  if (a_[axis] > b_[axis]) {
    swap(a_, b_);
  }
  const int a = a_[axis];
  const int b = b_[axis];

  bool origin = true;
  int left, right;
  if (axis == 0) {
    if (min(a_[oaxis], b_[oaxis]) == state->origin[oaxis]) {
      left = 0;
      right = 1;
    } else {
      origin = false;
      left = 2;
      right = 3;
    }
  } else {
    if (min(a_[oaxis], b_[oaxis]) == state->origin[oaxis]) {
      left = 0;
      right = 2;
    } else {
      origin = false;
      left = 1;
      right = 3;
    }
  }

  // The current cell is already subdivided once.
  int s = state->origin[axis] + (state->w>>1);
  while ((s <= a || s >= b) && (state->w>>1) > 1) {
    int position;
    if (s <= a) {
      s = s + (state->w>>2);
      position = right;
    } else {
      s = s - (state->w>>2);
      position = left;
    }
    split(state, position);
  }

  cout << "Split found between a = " << a << " and b = " << b
       << ": s = " << s << endl;
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
  state->indexStack.pop_back();
  (state->positionStack) >>= 2;
  state->center = make_intn(state->origin[0] + (state->w>>1),
                            state->origin[1] + (state->w>>1));
}

struct Intercept {
  // Point of intersection
  intn p;
  // Regions of intersection: 0, 1, 2, or 3.
  int region;
};

// axis - the axis. If axis = 0, then we're looking for the intersection
//        of the line with the hyperplane at x = c.
Intercept getIntercept(const WalkState* state,
    const int y_value, const int y_axis, const intn p0, const floatn v) {
  const int x_axis = 1 - y_axis;
  const float t = (y_value - p0[y_axis]) / v[y_axis];
  Intercept ret;
  ret.p =  p0 + convert_intn(v * t);
  ret.region = get_region(ret.p[x_axis],
                          state->center[x_axis],
                          (state->w>>1));
  return ret;
}

struct CellIntercepts {
  // Three y intercepts: origin, center, and top. One x intercept: center.
  // Intercept intercepts[4];
  Intercept ybottom;
  Intercept ycenter;
  Intercept ytop;
  Intercept xcenter;
};

void getIntercepts(
    const WalkState* state,
    const int axis, const intn p0, const floatn v,
    CellIntercepts* intercepts) {
  
  const int x_axis = 1-axis;
  const int y_axis = axis;

  intercepts->ybottom = getIntercept(
      state, state->origin[y_axis], y_axis, p0, v);
  intercepts->ycenter = getIntercept(
      state, state->origin[y_axis] + (state->w >> 1), y_axis, p0, v);
  intercepts->ytop = getIntercept(
      state, state->origin[y_axis] + state->w, y_axis, p0, v);
  intercepts->xcenter = getIntercept(
      state, state->origin[x_axis] + (state->w >> 1), x_axis, p0, v);


  // const int y[] = { state->origin[y_axis],
  //                   state->origin[y_axis] + (state->w >> 1),
  //                   state->origin[y_axis] + state->w };
  // for (int i = 0; i < 3; ++i) {
  //   const float t = (y[i] - p0[y_axis]) / v[y_axis];
  //   intercepts->intercepts[i] = getIntercept(
  //       state, state->center[y_axis], y_axis, p0, v);
  //   // intercepts->y[i] = p0 + convert_intn(v * t);
  //   // intercepts->regions[i] = get_region(intercepts->y[i][x_axis],
  //   //                                     state->center[x_axis],
  //   //                                     (state->w>>1));
  // }
  // // intercepts->center = getIntercept(state->center[x_axis], x_axis,
  // //                                   p0, v);
  // intercepts->intercepts[4] = getIntercept(
  //     state, state->center[x_axis], x_axis,
  //     p0, v);
}

// Contains subdivide tasks for between 0 and 2^DIM subdivisions
struct ToSubdivide {
  int w;
  // The number of subdivisions to do. Must be no greater than 2^DIM.
  int total;
  // The number of subdivisions complete.
  int i;
  // The subdivision quadrants.
  int quadrants[1<<DIM];
  // The points intersecting in the regions.
  intn a_points[1<<DIM];
  intn b_points[1<<DIM];
};

void addSubdivision(ToSubdivide* ret, const int quadrant,
                    const intn* a, const intn* b) {
  const int i = ret->total;
  ret->quadrants[i] = quadrant;
  ret->a_points[i] = *a;
  ret->b_points[i] = *b;
  ret->total++;
}

int getIntersectedQuadrants(const CellIntercepts* intercepts) {
  int ret = 0;
  // ybottom
  if (intercepts->ybottom.region == 1) {
    ret |= (1 << 0);
  } else if (intercepts->ybottom.region == 2) {
    ret |= (1 << 1);
  }
  // ycenter
  if (intercepts->ycenter.region == 1) {
    ret |= (1 << 0);
    ret |= (1 << 2);
  } else if (intercepts->ycenter.region == 2) {
    ret |= (1 << 1);
    ret |= (1 << 3);
  }
  // ytop
  if (intercepts->ytop.region == 1) {
    ret |= (1 << 2);
  } else if (intercepts->ytop.region == 2) {
    ret |= (1 << 3);
  }
  // xcenter
  if (intercepts->xcenter.region == 1) {
    ret |= (1 << 0);
    ret |= (1 << 1);
  } else if (intercepts->xcenter.region == 2) {
    ret |= (1 << 2);
    ret |= (1 << 3);
  }

  return ret;
}

ToSubdivide createToSubdivide(
    const WalkState* state,
    CellIntercepts* a, CellIntercepts* b) {
  // // Make sure a starts in quadrant 0
  // if (a->ybottom.region != 1) {
  //   swap(a, b);
  // }
  // if (a->ybottom.region != 1) {
  //   cout << "Illegal region" << endl;
  //   return ToSubdivide();//throw logic_error("Illegal region");
  // }

  ToSubdivide ret;
  ret.w = state->w;
  ret.total = 0;
  ret.i = 0;
  const int a_quads = getIntersectedQuadrants(a);
  const int b_quads = getIntersectedQuadrants(b);
  const int conflicts = a_quads & b_quads;
  for (int i = 0; i < NUM_CELLS; ++i) {
    if (conflicts & (1 << i)) {
      addSubdivision(&ret, i, &a->ybottom.p, &b->ybottom.p);
    }
  }

  // if (a->ybottom.region == 1 && b->ybottom.region == 2) {
  //   // quadrant 0
  //   if (b->xcenter.region == 1) {
  //     addSubdivision(&ret, 0, &a->ybottom.p, &b->xcenter.p);
  //   }
  //   // quadrant 1
  //   if (a->xcenter.region == 1) {
  //     addSubdivision(&ret, 1, &a->xcenter.p, &b->ybottom.p);
  //   }
  //   // quadrant 2
  //   if (a->ycenter.region == 1 && b->ycenter.region == 1) {
  //     addSubdivision(&ret, 2, &a->ycenter.p, &b->ycenter.p);
  //   }
  //   if (a->ycenter.region == 1 && b->xcenter.region == 2) {
  //     addSubdivision(&ret, 2, &a->ycenter.p, &b->xcenter.p);
  //   }
  //   // quadrant 3
  //   if (a->ycenter.region == 2 && b->ycenter.region == 2) {
  //     addSubdivision(&ret, 3, &a->ycenter.p, &b->ycenter.p);
  //   }
  //   if (a->xcenter.region == 2 && b->ycenter.region == 2) {
  //     addSubdivision(&ret, 3, &a->xcenter.p, &b->ycenter.p);
  //   }
  // }
  // // quadrant 0
  // if (a->ybottom.region == 1 && b->ybottom.region == 1) {
  //   addSubdivision(&ret, 0, &a->ybottom.p, &b->ybottom.p);
  // }
  // if (a->ycenter.region == 1 && b->ycenter.region == 1) {
  //   addSubdivision(&ret, 2, &a->ycenter.p, &b->ycenter.p);
  // }
  return ret;
}

ToSubdivide createToSubdivide(
    const WalkState* state, const int oaxis,
    const intn a_p0, const floatn a_v,
    const intn b_p0, const floatn b_v) {
  CellIntercepts a_intercepts, b_intercepts;
  getIntercepts(state, oaxis, a_p0, a_v, &a_intercepts);
  getIntercepts(state, oaxis, b_p0, b_v, &b_intercepts);
  return createToSubdivide(
      state, &a_intercepts, &b_intercepts);
}

// non-const for swap
vector<OctNode> fit(intn a_p0, floatn a_v,
                    intn b_p0, floatn b_v,
                    int w_) {
  cout << "--------------------------------------------------" << endl;
  cout << "fit" << endl;
  cout << "--------------------------------------------------" << endl;

  Resln resln(1<<options.max_level);

  WalkState state = createWalkState(w_);

  // int ax = a_p0.x;
  // int bx = b_p0.x;
  // int ay = a_p0.y;
  // int by = b_p0.y;
  // const int axis = (abs(ax-bx) > abs(ay-by)) ? 0 : 1;
  const int axis = 0;//(abs(a_v[0]) < abs(a_v[1])) ? 0 : 1;
  // oaxis is the walk axis
  const int oaxis = 1-axis;
  const int a_ = a_p0[axis];
  const int b_ = b_p0[axis];
  // Make sure a < b.
  if (a_ > b_) {
    swap(a_p0, b_p0);
    swap(a_v, b_v);
  }

  const bool negativeDir = (a_v[oaxis] < 0);

  // Do the initial split.
  split(&state, -1);
  // find_split(a_p0, b_p0, &state);

  vector<int> a_regions, b_regions;

  vector<ToSubdivide> toSubdivideStack;

  ToSubdivide toSubdivide = createToSubdivide(&state, oaxis,
                                              a_p0, a_v, b_p0, b_v);
  toSubdivideStack.push_back(toSubdivide);
  // Walk along the lines. If they ever both cross the same quadtree edge then
  // separate by subdividing. Call the edge a conflict edge. Two lines
  // crossing a conflict edge will then enter a conflict cell.
  // while ((state.w>>1) < resln.width) {
  while (!toSubdivideStack.empty()) {
    // CellIntercepts a_intercepts, b_intercepts;
    // getIntercepts(&state, oaxis, a_p0, a_v, &a_intercepts);
    // const int a_region = a_intercepts.ycenter.region;
    // const intn a_p = a_intercepts.ycenter.p;
    // getIntercepts(&state, oaxis, b_p0, b_v, &b_intercepts);
    // // const int b_region = intercepts[1].regions[1];
    // // const intn b_p = intercepts[1].y[1];
    // const int b_region = b_intercepts.ycenter.region;
    // const intn b_p = b_intercepts.ycenter.p;

    // a_regions.push_back(a_region);
    // b_regions.push_back(b_region);

    // ToSubdivide toSubdivide = createToSubdivide(
    //     &state, &a_intercepts, &b_intercepts);
    toSubdivide = toSubdivideStack.back();
    toSubdivideStack.pop_back();
    if (toSubdivide.i < toSubdivide.total) {
      const int i = toSubdivide.i++;
      toSubdivideStack.push_back(toSubdivide);
      const int quadrant = toSubdivide.quadrants[i];
      if (!isSplit(&state, quadrant)) {
        split(&state, quadrant);
        // find_split(toSubdivide.a_points[i],
        //            toSubdivide.b_points[i],
        //            &state);
      }

      toSubdivide = createToSubdivide(&state, oaxis,
                                      a_p0, a_v, b_p0, b_v);
      toSubdivideStack.push_back(toSubdivide);
    } else {
      // while (state.w < resln.width) {
      //   popLevel(&state);
      // }
      // // break;
      popLevel(&state);
    }

    // cout << endl;
    // cout << "checking center = " << state.center << endl;
    // // cout << "a = " << a
    // //      << " b = " << b << endl;

    // if (a_region == b_region && state.w > 1) {
    //   // The region is the same -- this is a conflict edge.
    //   // Subdivide the cell bordered by the conflict edge.
    //   cout << "conflict at " << a_region << endl;
    //   if (a_region == 0 || a_region == 3) {
    //     cout << "Unsupported region" << endl;
    //     return state.nodes;
    //   }

    //   if (a_region == 1 && !isSplit(&state, 0)) {
    //     split(&state, 0);
    //   } else if (a_region == 2 && !isSplit(&state, 1)) {
    //     split(&state, 1);
    //   } else {
    //     // Which quadrant the cell to subdivide is in.
    //     int position = (a_region == 1) ? (1<<oaxis) : 3;
    //     if (negativeDir) {
    //       position = (a_region == 1) ? 0 : (1<<axis);
    //     }
    //     // Do the initial split.
    //     split(&state, position);
    //     find_split(a_p, b_p, &state);
    //   }
    // } else if (a_region > 1 && b_region > 1 && !isSplit(&state, 1) && state.w > 1) {
    //   // Which quadrant the cell to subdivide is in.
    //   // int position = (a_region == 1) ? (1<<oaxis) : 3;
    //   // if (negativeDir) {
    //   //   position = (a_region == 1) ? 0 : (1<<axis);
    //   // }
    //   int position = 1;

    //   // const intn a_p = getIntercept(state.center[axis], axis, a_p0, a_v);
    //   // Do the initial split.
    //   split(&state, position);
    //   // find_split(a_p, b_p, &state);
    // } else {
    //   int position = getPosition(&state);
    //   // Loop until the cell is on the bottom (if axis = x) or cell is
    //   // on the left (if axis = y).
    //   // while (state.w<resln.width &&
    //   //        (negativeDir?~position:position) & (1<<oaxis)) {
    //   while (state.w<resln.width) {
    //     popLevel(&state);
    //     position = getPosition(&state);
    //   }

    //   popLevel(&state);
    //   position = getPosition(&state);

    // }
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
