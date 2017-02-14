#ifndef __OCTREE_2_H__
#define __OCTREE_2_H__

#include "clfw.hpp"
#include "Shaders/Shaders.hpp"
#include "Polylines/Polylines.h"
#include "GLUtilities/gl_utils.h"
#include "Options/Options.h"
#include "Kernels/Kernels.h"
#include "BoundingBox/BoundingBox.h"
#include <glm/glm.hpp>

extern "C" {
#include "OctreeResolution/Resln.h"
#include "Line/Line.h"
}
#include "Octree/OctNode.h"
#include "Quantize/Quantize.h"
#include "GLUtilities/Sketcher.h"

class Quadtree {
private:
  std::vector<OctNode> octree;
  std::vector<floatn> points;
	std::vector<cl_int> pointColors;
  std::vector<intn> quantized_points;
  std::vector<intn> resolutionPoints;
  std::vector<Line> lines;
  std::vector<Line> orderedLines;
  std::vector<Conflict> conflicts;
  std::vector<Leaf> leaves;
  int octreeSize;
  BoundingBox bb;
  Resln resln;
  int totalPoints;
  int totalLeaves;
  int totalResPoints;

  std::vector<glm::vec3> offsets;
  std::vector<glm::vec3> colors;
  std::vector<float> scales;

  GLuint boxProgram_vao;
  GLuint positions_vbo;
  GLuint position_indices_vbo;
  GLuint instance_vbo;
	cl::Buffer pointsBuffer;
	cl::Buffer pntColorsBuffer;
  cl::Buffer qpoints;
  cl::Buffer zpoints;
  cl::Buffer zpointsCopy;
  cl::Buffer linesBuffer;
  cl::Buffer resQPoints;
  cl::Buffer leavesBuffer;
  cl::Buffer octreeBuffer;

	void getPoints(const PolyLines *polyLines, vector<floatn> &points, vector <cl_int> &pointColors, std::vector<Line> &lines);
  void getBoundingBox(const vector<floatn> &points, const int totalPoints, BoundingBox &bb);

public:
  Quadtree();
	void build(const PolyLines* lines);
	void build(vector<floatn> &points, vector<cl_int> &pointColors, vector<Line> &lines, BoundingBox bb);
  typedef struct {
    float offset[3];
    float scale;
    float color[3];
  } Instance;
  std::vector<Instance> gl_instances;

  /* Drawing Methods */
  void draw(const glm::mat4& mvMatrix);

private:
	void build_internal();

  void clear();
  cl_int placePointsOnCurve(cl::Buffer points_i, int totalPoints, Resln resln, BoundingBox bb, string uniqueString, cl::Buffer &qpoints_o, cl::Buffer &zpoints_o);
	cl_int buildVertexOctree(cl::Buffer points_i, int totalPoints, Resln resln, BoundingBox bb, string uniqueString, cl::Buffer &octree_o, cl_int &totalOctnodes_o, cl::Buffer &leaves_o, cl_int &totalLeaves_o);
	cl_int buildPrunedOctree(cl::Buffer points_i, cl::Buffer pntColors_i, int totalPoints, Resln resln, BoundingBox bb, string uniqueString, cl::Buffer &octree_o, cl_int &totalOctnodes_o, cl::Buffer &leaves_o, cl_int &totalLeaves_o);
  cl_int resolveAmbiguousCells(cl::Buffer &octree_i, cl_int &totalOctNodes, cl::Buffer leaves_i, cl_int totalLeaves, cl::Buffer lines_i, cl_int totalLines, cl::Buffer qpoints_i, cl::Buffer zpoints_i, cl::Buffer pntCols_i, cl_int totalPoints, cl_int iteration);
  /* Drawing Methods */
  void addOctreeNodes(cl::Buffer octree, cl_int totalOctNodes);
  void addOctreeNodes(vector<OctNode> &octree, int index, floatn offset, float scale, float3 color);
  void addLeaf(vector<OctNode> octree, int internalIndex, int leafIndex, float3 color);
  void addConflictCells(cl::Buffer conflicts, cl::Buffer octree, cl_int totalOctnodes, cl::Buffer leaves, cl_int totalLeaves);
  void drawResolutionPoints(cl::Buffer resPoints, cl_int totalPoints);
};


#endif
