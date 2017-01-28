// Had to add this here to avoid compilation error on Mac.
#include <GL/glew.h>

#include "data.h"

namespace Data {
  PolyLines *lines;
  Quadtree *octree;
};
