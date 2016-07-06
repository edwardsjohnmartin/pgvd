#ifndef __OCTREE_2_H__
#define __OCTREE_2_H__

#include "LinesProgram.h"
#include "Polylines.h"
#include "gl_utils.h"
#include "CellIntersections.h"
#include "OctreeUtils.h"
#include "BoundingBox.h"
#include "Options.h"
#include "Resln.h"
#include "OctNode.h"

class Octree2 {
 private:
  std::vector<OctNode> octree;
  std::vector<CellIntersections> cell_intersections;
  std::vector<floatn> intersections;
  std::vector<floatn> karras_points;
  std::vector<intn> extra_qpoints;
  BoundingBox<float2> bb;
  vector<floatn> _origins;
  vector<float> _lengths;
  Resln resln;

  GLuint drawIndices[100];
  int numIndices;
  std::vector<glm::vec3> vertices;
  // int numVertices;
  GLuint drawVertices_vbo;
  GLuint drawIndices_vbo;
  GLuint vao;

 public:
  Octree2();
  
  int processArgs(int argc, char** argv);
  void build(const std::vector<float2>& points,
             const BoundingBox<float2>* customBB);
  void build(const PolyLines& lines, const BoundingBox<float2>* customBB = 0);
  void render(LinesProgram* program);
  void renderNode(LinesProgram* program, BigUnsigned lcp, int lcpLength);
  void renderBoundingBox(LinesProgram* program, const PolyLines& lines);

  void set(std::vector<OctNode>& octree_, const BoundingBox<float2>& bb_) {
    octree = octree_;
    bb = bb_;
    buildOctVertices();
  }

 private:
  float2 obj2Oct(const float2& v) const;
  float2 oct2Obj(const int2& v) const;
  GLfloat oct2Obj(int dist) const;
  glm::vec3 toVec3(float2 p) const;

  void Find(const float2& p);
  void FindMultiCells(const PolyLines& lines);

  void buildOctVertices();
  void drawNode(
    const OctNode& parent, const int parent_idx,
    const intn origin, const int length);


  std::vector<OctreeUtils::CellIntersection> Walk(
      const floatn& a, const floatn& b);

};


#endif
