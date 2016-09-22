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
  //std::vector<floatn> intersections;
  std::vector<float2> karras_points;
  std::vector<intn> qpoints;
  std::vector<Line> lines;
  std::vector<intn> extra_qpoints;
  BoundingBox bb;
  //vector<floatn> _origins;
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
  //void build(const std::vector<floatn>& points, const BoundingBox* customBB);
  void build(const PolyLines* lines);
  typedef struct {
    float offset[3];
    float scale;
    float color[3];
  } Instance;
  std::vector<Instance> instances;
  //void getZPoints(vector<BigUnsigned> &zpoints, const std::vector<intn> &qpoints);
  //int getNode(BigUnsigned lcp, int lcpLength, OctNode *octree);

  //void set(std::vector<OctNode>& octree_, const BoundingBox<floatn>& bb_) {
  //  octree = octree_;
  //  bb = bb_;
  //  buildOctVertices();
  //}

  /* Drawing Methods */
  void draw();

 private:
  /*floatn obj2Oct(const floatn& v) const;
  floatn oct2Obj(const intn& v) const;
  GLfloat oct2Obj(int dist) const;
  glm::vec3 toVec3(floatn p) const;

  */
  
  /* Drawing Methods */
  void addOctreeNodes();
  void addOctreeNodes(int index, floatn offset, float scale, float3 color);
  void addLeaf(int internalIndex, int leafIndex, float3 color);
  void findAmbiguousCells();
   /*
  void addNode(BigUnsigned lcp, int lcpLength, float colorStrength = 0.0);
  */
  /* Unused */
  //void FindMultiCells(const PolyLines& lines);
  //void Find(const floatn& p);
  //std::vector<OctreeUtils::CellIntersection> Walk(const floatn& a, const floatn& b);

};


#endif
