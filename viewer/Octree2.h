#ifndef __OCTREE_2_H__
#define __OCTREE_2_H__

#include "clfw.hpp"
#include "LinesProgram.h"
#include "Shaders.hpp"
#include "Polylines.h"
#include "gl_utils.h"
#include "CellIntersections.h"
//#include "OctreeUtils.h"
#include "BoundingBox.h"
#include "Options.h"

extern "C" {
#include "Resln.h"
#include "OctNode.h"
#include "Line.h"
}

class Octree2 {
 private:
  std::vector<OctNode> octree;
  std::vector<CellIntersections> cell_intersections;
  std::vector<floatn> intersections;
  std::vector<floatn> karras_points;
  std::vector<intn> qpoints;
  std::vector<Line> lines;
  std::vector<intn> extra_qpoints;
  BoundingBox<float2> bb;
  vector<floatn> _origins;
  vector<float> _lengths;
  Resln resln;

  GLuint drawIndices[100];
  int numIndices;
  std::vector<glm::vec3> vertices;
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
  
  int processArgs(int argc, char** argv);
  void build(const std::vector<float2>& points,
             const BoundingBox<float2>* customBB);
  void build(const PolyLines& lines, const BoundingBox<float2>* customBB = 0);
  typedef struct {
    float offset[3];
    float scale;
    float color[3];
  } Instance;
  std::vector<Instance> instances;
  void getZPoints(vector<BigUnsigned> &zpoints, const std::vector<intn> &qpoints);
  int getNode(BigUnsigned lcp, int lcpLength, OctNode *octree);

  //void set(std::vector<OctNode>& octree_, const BoundingBox<float2>& bb_) {
  //  octree = octree_;
  //  bb = bb_;
  //  buildOctVertices();
  //}

  /* Drawing Methods */
  void draw();

 private:
  float2 obj2Oct(const float2& v) const;
  float2 oct2Obj(const int2& v) const;
  GLfloat oct2Obj(int dist) const;
  glm::vec3 toVec3(float2 p) const;


  
  /* Drawing Methods */
  void addOctreeNodes();
  void addOctreeNodes(int index, float2 offset, float scale, float3 color);
  void findAmbiguousCells();
  void addNode(BigUnsigned lcp, int lcpLength, float colorStrength = 0.0);

  /* Unused */
  //void FindMultiCells(const PolyLines& lines);
  void Find(const float2& p);
  //std::vector<OctreeUtils::CellIntersection> Walk(const floatn& a, const floatn& b);

};


#endif
