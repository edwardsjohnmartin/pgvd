#ifndef __OCTREE_2_H__
#define __OCTREE_2_H__

#include "clfw.hpp"
#include "../Shaders/Shaders.hpp"
#include "../Polylines/Polylines.h"
#include "../GLUtilities/gl_utils.h"
#include "../Options/Options.h"
#include "../Kernels/Kernels.h"
#include <glm/glm.hpp>

extern "C" {
#include "../../C/BoundingBox/BoundingBox.h"
#include "../../C/OctreeResolution/Resln.h"
#include "../../C/Octree/OctNode.h"
#include "../../C/Line/Line.h"
}

class Octree2 {
 private:
  std::vector<OctNode> octree;
  std::vector<float_2> karras_points;
  std::vector<int_n> quantized_points;
  std::vector<Line> lines;
  BoundingBox bb;
  Resln resln;

  std::vector<glm::vec3> offsets;
  std::vector<glm::vec3> colors;
  std::vector<float> scales;

  GLuint drawVertices_vbo;
  GLuint drawIndices_vbo;
  GLuint vao;

  GLuint boxProgram_vao;
  GLuint positions_vbo;
  GLuint position_indices_vbo;
  GLuint instance_vbo;

 public:
  Octree2();
  
  void build(const PolyLines* lines);
  typedef struct {
    float offset[3];
    float scale;
    float color[3];
  } Instance;
  std::vector<Instance> gl_instances;

  /* Drawing Methods */
  void draw();

 private:
 
  /* Drawing Methods */
  void addOctreeNodes();
  void addOctreeNodes(int index, float_n offset, float scale, float_3 color);
  void addLeaf(int internalIndex, int leafIndex, float_3 color);
};


#endif
