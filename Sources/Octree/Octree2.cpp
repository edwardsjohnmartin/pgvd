#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <ctime>
#include <chrono>

#include "GLUtilities/gl_utils.h"
#include "./Octree2.h"
#include "Catch/HelperFunctions.hpp"

using namespace std;
using namespace Kernels;

#define benchmark(text)
#define check(error) {assert_cl_error(error);}

Quadtree::Quadtree() {
  const int n = 4;

  // Octree drawn using instanced, indexed rendered cubes.
  glGenBuffers(1, &positions_vbo);
  glGenBuffers(1, &instance_vbo);
  glGenBuffers(1, &position_indices_vbo);
  glGenVertexArrays(1, &boxProgram_vao);
  glBindVertexArray(boxProgram_vao);
  glEnableVertexAttribArray(Shaders::boxProgram->position_id);
  glEnableVertexAttribArray(Shaders::boxProgram->offset_id);
  glEnableVertexAttribArray(Shaders::boxProgram->scale_id);
  glEnableVertexAttribArray(Shaders::boxProgram->color_id);
  print_gl_error();
  float points[] = {
    -.5, -.5, -.5,  -.5, -.5, +.5,
    +.5, -.5, +.5,  +.5, -.5, -.5,
    -.5, +.5, -.5,  -.5, +.5, +.5,
    +.5, +.5, +.5,  +.5, +.5, -.5,
  };
  unsigned char indices[] = {
    0, 1, 1, 2, 2, 3, 3, 0,
    0, 4, 1, 5, 2, 6, 3, 7,
    4, 5, 5, 6, 6, 7, 7, 4,
  };
  glBindBuffer(GL_ARRAY_BUFFER, positions_vbo);
  print_gl_error();
  glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
  print_gl_error();
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, position_indices_vbo);
  fprintf(stderr, "position_indices_vbo: %d\n", position_indices_vbo);
  print_gl_error();
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  print_gl_error();
  glVertexAttribPointer(Shaders::boxProgram->position_id, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
  print_gl_error();
  glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);
  print_gl_error();
  glVertexAttribPointer(Shaders::boxProgram->offset_id, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), 0);
  print_gl_error();
  glVertexAttribPointer(Shaders::boxProgram->scale_id, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
  print_gl_error();
  glVertexAttribPointer(Shaders::boxProgram->color_id, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(4 * sizeof(float)));
  print_gl_error();
  glVertexAttribDivisor(Shaders::boxProgram->offset_id, 1);
  print_gl_error();
  glVertexAttribDivisor(Shaders::boxProgram->scale_id, 1);
  print_gl_error();
  glVertexAttribDivisor(Shaders::boxProgram->color_id, 1);
  glBindVertexArray(0);
  print_gl_error();

  resln = make_resln(1 << Options::max_level);
}

cl_int Quadtree::placePointsOnCurve(
	cl::Buffer points_i, 
	int totalPoints, 
	Resln resln, 
	BoundingBox bb, 
	string uniqueString, 
	cl::Buffer &qpoints_o, 
	cl::Buffer &zpoints_o) {
  using namespace Kernels;
  cl_int error = 0;

  /* Quantize the points. */
  error |= QuantizePoints_p(points_i, totalPoints, bb, resln.width, uniqueString, qpoints_o);
  check(error);

  /* Convert the points to Z-Order */
  error |= QPointsToZPoints_p(qpoints_o, totalPoints, resln.bits, uniqueString, zpoints_o);
  check(error);

  return error;
}

cl_int Quadtree::buildVertexOctree(
	cl::Buffer zpoints_i, 
	int totalPoints, 
	Resln resln, 
	BoundingBox bb, 
	string uniqueString, 
	cl::Buffer &octree_o, 
	cl_int &totalOctnodes_o, 
	cl::Buffer &leaves_o, 
	cl_int &totalLeaves_o) 
{
  using namespace Kernels;
  cl_int error = 0;
  cl_int uniqueTotalPoints = totalPoints;
  cl::Buffer zpoints_copy, brt, nullBuffer;
	/* Make a copy of the zpoints. */
	CLFW::get(zpoints_copy, uniqueString + "zptscpy", nextPow2(sizeof(BigUnsigned) * totalPoints));
	error |= CLFW::DefaultQueue.enqueueCopyBuffer(zpoints_i, zpoints_copy, 0, 0, totalPoints * sizeof(BigUnsigned));
	check(error);

  /* Radix sort the zpoints */
  error |= RadixSortBigUnsigned_p(zpoints_copy, totalPoints, resln.mbits, uniqueString);
  check(error);
	
  /* Unique the zpoints */
  error |= UniqueSorted(zpoints_copy, totalPoints, uniqueString, uniqueTotalPoints);
  check(error);

  /* Build a binary radix tree*/
  error |= BuildBinaryRadixTree_p(zpoints_copy, uniqueTotalPoints, resln.mbits, uniqueString, brt);
  check(error);

  /* Convert the binary radix tree to an octree*/
  error |= BinaryRadixToOctree_p(brt, false, nullBuffer, uniqueTotalPoints, uniqueString, octree_o, totalOctnodes_o); //occasionally currentSize is 0...
  check(error);

  /* Use the internal octree nodes to calculate leaves */
  error |= GetLeaves_p(octree_o, totalOctnodes_o, leaves_o, totalLeaves_o);
  check(error);

  return error;
}

cl_int Quadtree::buildPrunedOctree(
	cl::Buffer zpoints_i,
	cl::Buffer pntColors_i,
	int totalPoints,
	Resln resln,
	BoundingBox bb,
	string uniqueString,
	cl::Buffer &octree_o,
	cl_int &totalOctnodes_o,
	cl::Buffer &leaves_o,
	cl_int &totalLeaves_o)
{
	using namespace Kernels;
	cl_int error = 0;
	cl_int uniqueTotalPoints = totalPoints;
	cl::Buffer zpoints_copy, pntColors_copy, brt, brtColors;
	/* Make a copy of the zpoints and colors. */
	CLFW::get(zpoints_copy, uniqueString + "zptscpy", sizeof(BigUnsigned) * nextPow2(totalPoints));
	error |= CLFW::DefaultQueue.enqueueCopyBuffer(zpoints_i, zpoints_copy, 0, 0, totalPoints * sizeof(BigUnsigned));

	CLFW::get(pntColors_copy, uniqueString + "colscpy", sizeof(cl_int) * nextPow2(totalPoints));
	error |= CLFW::DefaultQueue.enqueueCopyBuffer(pntColors_i, pntColors_copy, 0, 0, sizeof(cl_int) * totalPoints);

	/* Radix sort the zpoints */
	error |= RadixSortBUIntPairsByKey(zpoints_copy, pntColors_copy, resln.mbits, totalPoints);
	check(error);

	/* Unique the zpoints */
	error |= UniqueSortedBUIntPair(zpoints_copy, pntColors_copy, totalPoints, uniqueString, uniqueTotalPoints);
	check(error);
	
	/* Build a colored binary radix tree*/
	error |= BuildColoredBinaryRadixTree_p(zpoints_copy, pntColors_copy, uniqueTotalPoints, resln.mbits, uniqueString, brt, brtColors);
	check(error);

	/* Identify required cells */
	error |= PropagateBRTColors_p(brt, brtColors, uniqueTotalPoints - 1, uniqueString);
	check(error);

	/* Convert the binary radix tree to an octree*/
	error |= BinaryRadixToOctree_p(brt, true, brtColors, uniqueTotalPoints, uniqueString, octree_o, totalOctnodes_o); //occasionally currentSize is 0...
	check(error);

	/* Use the internal octree nodes to calculate leaves */
	error |= GetLeaves_p(octree_o, totalOctnodes_o, leaves_o, totalLeaves_o);
	check(error);
	
	return error;
}

static cl_int InitializeConflictCellDetection(
	cl::Buffer &zpoints_i, 
	cl::Buffer &lines_i, 
	cl_int numLines,
	Resln &resln,
	cl::Buffer &octree_i,
	cl_int numOctNodes,
	cl::Buffer &lineIndices_o,
	cl::Buffer &LCPBounds_o
) {
	using namespace Kernels;
	cl_int error = 0;
	/* Compute line bounding cells and generate the unordered line indices. */
	cl::Buffer LineLCPs;
	error |= GetLineLCPs_p(lines_i, numLines, zpoints_i, resln.mbits, LineLCPs);
	error |= InitializeFacetIndices_p(numLines, lineIndices_o);
	check(error);

	/* For each bounding cell, look up it's surrounding octnode in the tree. */
	cl::Buffer LCPToOctNode;
	error |= LookUpOctnodeFromLCP_p(LineLCPs, numLines, octree_i, LCPToOctNode);
	check(error);

	/* Sort the node to line pairs by key. This gives us a node to facet mapping for conflict cell detection. */
	error |= RadixSortPairsByKey(LCPToOctNode, lineIndices_o, numLines);
	check(error);

	/* For each octnode, determine the first and last bounding cell index to be used for conflict cell detection. */
	error |= GetLCPBounds_p(LCPToOctNode, numLines, numOctNodes, LCPBounds_o);
	check(error);
	return error;
}

static cl_int FindConflictCells(
	cl::Buffer &octree_i,
	cl_int numOctNodes,
	cl::Buffer &leaves_i,
	cl_int numLeaves,
	cl::Buffer &qpoints_i,
	cl::Buffer &zpoints_i,
	cl::Buffer &lines_i,
	cl_int numLines,
	Resln &resln,
	cl::Buffer &conflicts_o,
	cl_int &numConflicts
	) {
	using namespace Kernels;
	cl_int error = 0;

	/* Initialize the node to facet mapping*/
	cl::Buffer LineLCPs, lineIndices, LCPBounds;
	error |= InitializeConflictCellDetection(zpoints_i, lines_i, numLines, resln,
		octree_i, numOctNodes, lineIndices, LCPBounds);

	/* Use that mapping to find conflict cells*/
	cl::Buffer sparseConflicts;
	error |= FindConflictCells_p(octree_i, leaves_i, numLeaves, lineIndices,
		LCPBounds, lines_i, numLines, qpoints_i, resln.width, sparseConflicts);
	check(error);

	/* Compact the non-conflict cells to the right */
	cl::Buffer cPred, cAddr;
	error |= CLFW::get(cPred, "cPred", sizeof(cl_int) * nextPow2(numLeaves));
	error |= CLFW::get(cAddr, "cAddr", sizeof(cl_int) * nextPow2(numLeaves));
	error |= PredicateConflicts_p(sparseConflicts, numLeaves, "", cPred);
	error |= StreamScan_p(cPred, numLeaves, "cnflctaddr", cAddr);
	error |= CLFW::Download<cl_int>(cAddr, numLeaves - 1, numConflicts);
	error |= CLFW::get(conflicts_o, "conflicts", sizeof(Conflict) * nextPow2(numLeaves));
	error |= CompactConflicts_p(sparseConflicts, cPred, cAddr, numLeaves, conflicts_o);
	check(error);

	return error;
}

static cl_int GenerateResolutionPoints(
	cl::Buffer &conflicts_i,
	cl_int numConflicts,
	cl::Buffer &qpoints_i,
	cl::Buffer &resPts,
	cl_int &numResPts
	) {
	using namespace Kernels;
	cl_int error = 0;

	/* Use the conflicts to initialize data required to calculate resolution points */
	cl::Buffer conflictInfo, numPtsPerConflict, scannedNumPtsPerConflict, predPntToConflict, pntToConflict;
	error |= GetResolutionPointsInfo_p(conflicts_i, numConflicts, qpoints_i, conflictInfo, numPtsPerConflict);
	check(error);

	/* Scan the pts per conflict to determine beginning and ending addresses for res pts*/
	error |= CLFW::get(scannedNumPtsPerConflict, "snptspercnflct", sizeof(cl_int) * nextPow2(numConflicts));
	error |= StreamScan_p(numPtsPerConflict, numConflicts, "conflictInfo", scannedNumPtsPerConflict);
	error |= CLFW::Download<cl_int>(scannedNumPtsPerConflict, numConflicts - 1, numResPts);
	check(error);

	/* Create a res pnt to conflict info mapping so we can determine resolution points in parallel. */
	error |= PredicatePointToConflict_p(scannedNumPtsPerConflict, numConflicts, numResPts, predPntToConflict);
	error |= CLFW::get(pntToConflict, "pnt2Conflict", nextPow2(sizeof(cl_int) * numResPts));
	error |= StreamScan_p(predPntToConflict, numResPts, "pnt2Conf", pntToConflict);
	check(error);

	/* Get the resolution points */
	error |= GetResolutionPoints_p(conflicts_i, conflictInfo, scannedNumPtsPerConflict, numResPts, pntToConflict, qpoints_i, resPts);
	check(error);

	return error;
}

static cl_int combinePoints(
	cl::Buffer &qpoints_i,
	cl::Buffer &zpoints_i,
	cl::Buffer &pntCols_i,
	cl_int numPts,
	cl::Buffer &resPts_i,
	cl::Buffer &resZPts_i,
	cl_int numResPts,
	cl_int iteration,
	cl::Buffer &combinedQPts_o,
	cl::Buffer &combinedZPts_o,
	cl::Buffer &combinedCols_o
	) {
	using namespace Kernels;
	cl_int error = 0;

	cl::CommandQueue &queue = CLFW::DefaultQueue;
	error |= CLFW::get(combinedQPts_o, "qpoints" + std::to_string(iteration), nextPow2(numPts + numResPts) * sizeof(intn));
	error |= queue.enqueueCopyBuffer(qpoints_i, combinedQPts_o, 0, 0, numPts * sizeof(intn));
	error |= queue.enqueueCopyBuffer(resPts_i, combinedQPts_o, 0, numPts * sizeof(intn), numResPts * sizeof(intn));
	check(error);

	error |= CLFW::get(combinedZPts_o, "zpoints" + std::to_string(iteration), nextPow2(numPts + numResPts) * sizeof(BigUnsigned));
	error |= queue.enqueueCopyBuffer(zpoints_i, combinedZPts_o, 0, 0, numPts * sizeof(BigUnsigned));
	error |= queue.enqueueCopyBuffer(resZPts_i, combinedZPts_o, 0, numPts * sizeof(BigUnsigned), numResPts * sizeof(BigUnsigned));
	check(error);

	error |= CLFW::get(combinedCols_o, "ptcol" + std::to_string(iteration), nextPow2(numPts + numResPts) * sizeof(cl_int));
	error |= queue.enqueueCopyBuffer(pntCols_i, combinedCols_o, 0, 0, numPts * sizeof(cl_int));
	vector<cl_int> resCols(numResPts);
	for (int i = 0; i < numResPts; ++i) resCols[i] = -3 - i;
	error |= queue.enqueueWriteBuffer(combinedCols_o, CL_TRUE, numPts * sizeof(cl_int), numResPts * sizeof(cl_int), resCols.data());
	check(error);
	return error;
}

cl_int Quadtree::resolveAmbiguousCells(
  cl::Buffer &octree_i, 
  cl_int &numOctNodes, 
  cl::Buffer leaves_i, 
  cl_int numLeaves, 
  cl::Buffer lines_i, 
  cl_int numLines, 
  cl::Buffer qpoints_i,
	cl::Buffer zpoints_i,
	cl::Buffer pntCols_i,
	cl_int numPts,
	cl_int iteration
) { 
	if (iteration > 1) return CL_SUCCESS;
	if (numLines <= 1) return CL_SUCCESS;
  using namespace Kernels;
  cl_int error = 0;

  /* Determine conflicts to resolve */
	cl::Buffer conflicts; cl_int numConflicts = 0;
	error |= FindConflictCells(octree_i, numOctNodes, leaves_i, numLeaves,
		qpoints_i, zpoints_i, lines_i, numLines, resln, conflicts, numConflicts);
	check(error);

	/* If there are no more conflicts to resolve, we're done. */
	if (numConflicts == 0) return error;

	/* Use the conflicts to generate resolution points */
	cl::Buffer resPts; cl_int numResPts;
	error |= GenerateResolutionPoints(conflicts, numConflicts, qpoints_i, resPts, numResPts);
	check(error);

	if (Options::showResolutionPoints) drawResolutionPoints(resPts, numResPts);

	/* Convert to z-order */
	cl::Buffer resZPoints;
	error |= QPointsToZPoints_p(resPts, numResPts, resln.bits, "res", resZPoints);
	check(error);

	/* Combine the original and generated resolution points */
	cl::Buffer combinedQPts, combinedZPts, combinedCols;
	combinePoints(qpoints_i, zpoints_i, pntCols_i, numPts,
		resPts, resZPoints, numResPts, iteration, combinedQPts, combinedZPts, combinedCols);
	check(error);
	
	/* Build an octree from the combined points */
	cl::Buffer combinedOctree, combinedLeaves;
	cl_int combinedOctSize, combinedLeafSize;
	error |= buildPrunedOctree(combinedZPts, combinedCols, numPts + numResPts, resln, bb, 
		"res" + iteration, combinedOctree, combinedOctSize, combinedLeaves, combinedLeafSize);

	octree_i = combinedOctree;
	numOctNodes = combinedOctSize;
	check(error);

	/* If the resolution points don't effect the octree, quit resolving. */
	if (combinedOctSize == numOctNodes) {
		return error;
	}

	/* resolve further conflicts */
	if (iteration < Options::maxConflictIterations)
		resolveAmbiguousCells(combinedOctree, combinedOctSize, combinedLeaves, combinedLeafSize, lines_i, numLines, combinedQPts,
			combinedZPts, combinedCols, numPts + numResPts, iteration + 1);

  return error;
}

void Quadtree::clear() {
  using namespace GLUtilities;
  Sketcher::instance()->clear();
  octreeSize = 0;
  totalResPoints = 0;

  points.clear();
	pointColors.clear();
  gl_instances.clear();
  octree.clear();
  resolutionPoints.clear();
}

void Quadtree::build(const PolyLines *polyLines) {
	bool resolveConflicts = false;

  using namespace Kernels;
  CLFW::DefaultQueue = CLFW::Queues[0];
  cl_int error = 0;

  /* Clear the old quadtree */
  clear();  

  /* Extract points from objects, and calculate a bounding box. */
  getPoints(polyLines, points, pointColors, lines);
	if (points.size() == 0) return;

  /* Upload the data to OpenCL buffers */
	error |= CLFW::get(pointsBuffer, "pts", Kernels::nextPow2(points.size()) * sizeof(floatn));
	error |= CLFW::get(pntColorsBuffer, "ptcolr", Kernels::nextPow2(points.size()) * sizeof(cl_int));
  error |= CLFW::get(linesBuffer, "lines", Kernels::nextPow2(lines.size())*sizeof(Line));
	check(error);
  
	error |= CLFW::Upload<floatn>(points, pointsBuffer);
	error |= CLFW::Upload<cl_int>(pointColors, pntColorsBuffer);
  error |= CLFW::Upload<Line>(lines, linesBuffer);
  check(error);

	getBoundingBox(points, points.size(), bb);

	/* Place the points on a Z-Order curve */
  error |= placePointsOnCurve(pointsBuffer, points.size(), resln, bb, "initial", qpoints, zpoints);
  check(error);

  /* Build the initial octree */
  CLFW::DefaultQueue = CLFW::Queues[0];
	error |= buildPrunedOctree(zpoints, pntColorsBuffer, points.size(), resln, bb, "initial", octreeBuffer, octreeSize, leavesBuffer, totalLeaves);
	check(error);

		/* Finally, resolve the ambiguous cells. */
	error |= resolveAmbiguousCells(octreeBuffer, octreeSize, leavesBuffer, totalLeaves, 
		linesBuffer, lines.size(), qpoints, zpoints, pntColorsBuffer, points.size(), 0);
	check(error);
  
  /* Add the octnodes and conflict cells so they'll be rendered with OpenGL. */
  addOctreeNodes(octreeBuffer, octreeSize);
}

inline floatn getMinFloat(const floatn a, const floatn b) {
  floatn result;
  for (int i = 0; i < DIM; ++i) {
    result.s[i] = (a.s[i] < b.s[i]) ? a.s[i] : b.s[i];
  }
  return result;
}

inline floatn getMaxFloat(const floatn a, const floatn b) {
  floatn result;
  for (int i = 0; i < DIM; ++i) {
    result.s[i] = (a.s[i] > b.s[i]) ? a.s[i] : b.s[i];
  }
  return result;
}

void Quadtree::getPoints(const PolyLines *polyLines, vector<floatn> &points, vector<cl_int> &pointColors, std::vector<Line> &lines) {
  benchmark("getPoints");

  const vector<vector<floatn>> polygons = polyLines->getPolygons();
  lines = polyLines->getLines();

  // Get all vertices into a 1D array.
  for (int i = 0; i < polygons.size(); ++i) {
    const vector<floatn>& polygon = polygons[i];
    for (int j = 0; j < polygon.size(); ++j) {
      points.push_back(polygon[j]);
			pointColors.push_back(i);
    }
  }
}

void Quadtree::getBoundingBox(const vector<floatn> &points, const int totalPoints, BoundingBox &bb) {
  benchmark("getBoundingBox");

  if (Options::xmin == -1 && Options::xmax == -1) {
    //Probably should be parallelized...
    floatn minimum = points[0];
    floatn maximum = points[0];
    for (int i = 1; i < totalPoints; ++i) {
      minimum = getMinFloat(points[i], minimum);
      maximum = getMaxFloat(points[i], maximum);
    }
    bb = BB_initialize(&minimum, &maximum);
    bb = BB_make_centered_square(&bb);
  }
  else {
    bb.initialized = true;
    bb.minimum = make_floatn(Options::xmin, Options::ymin);
    bb.maximum = make_floatn(Options::xmax, Options::ymax);
    bb.maxwidth = BB_max_size(&bb);
  }
}

/* Drawing Methods */
void Quadtree::addOctreeNodes(cl::Buffer octree, cl_int totalOctNodes) {
  floatn temp;
  floatn center;

  if (totalOctNodes == 0) return;
	vector<OctNode> octree_vec;
	CLFW::Download<OctNode>(octree, totalOctNodes, octree_vec);

  center = (bb.minimum + bb.maxwidth*.5);
  float3 color = { 0.75, 0.75, 0.75 };
  addOctreeNodes(octree_vec, 0, center, bb.maxwidth, color);
}

void Quadtree::addOctreeNodes(vector<OctNode> &octree, int index, floatn offset, float scale, float3 color)
{
  Instance i = { offset.x, offset.y, 0.0, scale, color.x, color.y, color.z };
  gl_instances.push_back(i);
  if (index != -1) {
    OctNode current = octree[index];
    scale /= 2.0;
    float shift = scale / 2.0;
    addOctreeNodes(octree, current.children[0], { offset.x - shift, offset.y - shift },
      scale, color);
    addOctreeNodes(octree, current.children[1], { offset.x + shift, offset.y - shift },
      scale, color);
    addOctreeNodes(octree, current.children[2], { offset.x - shift, offset.y + shift },
      scale, color);
    addOctreeNodes(octree, current.children[3], { offset.x + shift, offset.y + shift },
      scale, color);
  }
}

void Quadtree::addLeaf(vector<OctNode> octree, int internalIndex, int childIndex, float3 color) {
  floatn center = BB_center(&bb);
  float octreeWidth = bb.maxwidth;
  OctNode node = octree[internalIndex];

  //Shift to account for the leaf
  float width = octreeWidth / (1 << (node.level + 1));
  float shift = width / 2.0;
  center.x += (childIndex & 1) ? shift : -shift;
  center.y += (childIndex & 2) ? shift : -shift;

  //Shift for the internal node
  while (node.parent != -1) {
    for (childIndex = 0; childIndex < 4; ++childIndex) {
      if (octree[node.parent].children[childIndex] == internalIndex)
        break;
    }

    shift *= 2.0;
    center.x += (childIndex & 1) ? shift : -shift;
    center.y += (childIndex & 2) ? shift : -shift;

    internalIndex = node.parent;
    node = octree[node.parent];
  }

  Instance i = {
    { center.x, center.y, 0.0 }
    , width
    , { color.x, color.y, color.z }
  };
  gl_instances.push_back(i);
}

void Quadtree::addConflictCells(cl::Buffer sparseConflicts, cl::Buffer octree, cl_int totalOctnodes, cl::Buffer leaves, cl_int totalLeaves) {
  // if (Options::debug) {
	vector<Leaf> leaves_v;
	vector<Conflict> conflicts_v;
	vector<OctNode> octree_v;
  CLFW::Download<Conflict>(sparseConflicts, totalLeaves, conflicts_v);
  CLFW::Download<Leaf>(leaves, totalLeaves, leaves_v);
	CLFW::Download<OctNode>(octree, totalOctnodes, octree_v);
  if (conflicts_v.size() == 0) return;
  for (int i = 0; i < conflicts_v.size(); ++i) {
    if (conflicts_v[i].color == -2)
    {
      addLeaf(octree_v, leaves_v[i].parent, leaves_v[i].quadrant,
      { Options::conflict_color[0],
        Options::conflict_color[1],
        Options::conflict_color[2] });
    }
  }
}

void Quadtree::drawResolutionPoints(cl::Buffer resPoints, cl_int totalPoints) {
  using namespace GLUtilities;
  vector<intn> resolutionPoints;
  CLFW::Download<intn>(resPoints, totalPoints, resolutionPoints);
  for (int i = 0; i < resolutionPoints.size(); ++i) {
    floatn point = UnquantizePoint(&resolutionPoints[i], &bb.minimum, resln.width, bb.maxwidth);
    Point p = {
      {point.x, point.y, 0.0, 1.0},
      {0.0, 0.0, 1.0, 1.0}
    };
    Sketcher::instance()->add(p);
  }
}

void Quadtree::draw(const glm::mat4& mvMatrix) {
  Shaders::boxProgram->use();
  print_gl_error();
  glBindVertexArray(boxProgram_vao);
  print_gl_error();
  glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);
  print_gl_error();
  glBufferData(GL_ARRAY_BUFFER, sizeof(Instance) * gl_instances.size(), gl_instances.data(), GL_STREAM_DRAW);
  print_gl_error();
  glUniformMatrix4fv(Shaders::boxProgram->matrix_id, 1, 0, &(mvMatrix[0].x));
  glUniform1f(Shaders::boxProgram->pointSize_id, 10.0);
  print_gl_error();
  glLineWidth(2.0);
  ignore_gl_error();
  glDrawElementsInstanced(GL_LINES, 12 * 2, GL_UNSIGNED_BYTE, 0, gl_instances.size());
  print_gl_error();
  glBindVertexArray(0);
  print_gl_error();
}

#undef benchmark
#undef check