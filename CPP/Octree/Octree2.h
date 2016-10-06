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
  //std::vector<CellIntersections> cell_intersections;
  //std::vector<float_n> intersections;
  std::vector<float_2> karras_points;
  std::vector<int_n> quantized_points;
  std::vector<Line> lines;
  BoundingBox bb;
  //vector<float_n> _origins;
  //vector<float> _lengths;
  Resln resln;

  //GLuint drawIndices[100];
  //int numIndices;
  //std::vector<glm::vec3> vertices;
  std::vector<glm::vec3> offsets;
  std::vector<glm::vec3> colors;
  std::vector<float> scales;

  // int numVertices;
  GLuint drawVertices_vbo;
  GLuint drawIndices_vbo;
  GLuint vao;

  GLuint boxProgram_vao;
  GLuint positions_vbo;
  GLuint position_indices_vbo;
  GLuint instance_vbo;

 public:
  Octree2();
  
  //int processArgs(int argc, char** argv);
  //void build(const std::vector<float_n>& points, const BoundingBox* customBB);
  void build(const PolyLines* lines);
  typedef struct {
    float offset[3];
    float scale;
    float color[3];
  } Instance;
  std::vector<Instance> gl_instances;
  //void getZPoints(vector<BigUnsigned> &zpoints, const std::vector<int_n> &qpoints);
  //int getNode(BigUnsigned lcp, int lcpLength, OctNode *octree);

  //void set(std::vector<OctNode>& octree_, const BoundingBox<float_n>& bb_) {
  //  octree = octree_;
  //  bb = bb_;
  //  buildOctVertices();
  //}

  /* Drawing Methods */
  void draw();

 private:
  /*float_n obj2Oct(const float_n& v) const;
  float_n oct2Obj(const int_n& v) const;
  GLfloat oct2Obj(int dist) const;
  glm::vec3 toVec3(float_n p) const;

  */
  
  /* Drawing Methods */
  void addOctreeNodes();
  void addOctreeNodes(int index, float_n offset, float scale, float_3 color);
  void addLeaf(int internalIndex, int leafIndex, float_3 color);
  void findAmbiguousCells();
   /*
  void addNode(BigUnsigned lcp, int lcpLength, float colorStrength = 0.0);
  */
  /* Unused */
  //void FindMultiCells(const PolyLines& lines);
  //void Find(const float_n& p);
  //std::vector<OctreeUtils::CellIntersection> Walk(const float_n& a, const float_n& b);

};


#endif
