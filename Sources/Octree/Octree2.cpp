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

cl_int Quadtree::placePointsOnCurve(cl::Buffer points_i, int totalPoints, Resln resln, BoundingBox bb, string uniqueString, cl::Buffer &qpoints_o, cl::Buffer &zpoints_o) {
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
  cl::Buffer zpoints_copy, brt;
	/* Make a copy of the zpoints. */
	CLFW::get(zpoints_copy, uniqueString + "zptscpy", nextPow2(sizeof(BigUnsigned) * totalPoints));
	error |= CLFW::DefaultQueue.enqueueCopyBuffer(zpoints_i, zpoints_copy, 0, 0, totalPoints * sizeof(BigUnsigned));

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
  error |= BinaryRadixToOctree_p(brt, uniqueTotalPoints, uniqueString, octree_o, totalOctnodes_o); //occasionally currentSize is 0...
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
	cl::Buffer zpoints_copy, brt;
	/* Make a copy of the zpoints. */
	CLFW::get(zpoints_copy, uniqueString + "zptscpy", nextPow2(sizeof(BigUnsigned) * totalPoints));
	error |= CLFW::DefaultQueue.enqueueCopyBuffer(zpoints_i, zpoints_copy, 0, 0, totalPoints * sizeof(BigUnsigned));

	/* Radix sort the zpoints */
	error |= RadixSortBUIntPairsByKey(zpoints_copy, pntColors_i, totalPoints, resln.mbits);
	check(error);
	CLFW::DefaultQueue.finish();
	/* Unique the zpoints */
	error |= UniqueSortedBUIntPair(zpoints_copy, pntColors_i, totalPoints, uniqueString, uniqueTotalPoints);
	check(error);
	CLFW::DefaultQueue.finish();

	/* Build a binary radix tree*/
	error |= BuildBinaryRadixTree_p(zpoints_copy, uniqueTotalPoints, resln.mbits, uniqueString, brt);
	check(error);
	CLFW::DefaultQueue.finish();

	vector<BrtNode> brt_vec;
	//CLFW::Download<BrtNode>(brt, 1, brt_vec);
	//writeToFile<BrtNode>(brt_vec, "TestData//simple//brt.bin");

	/* Convert the binary radix tree to an octree*/
	error |= BinaryRadixToOctree_p(brt, uniqueTotalPoints, uniqueString, octree_o, totalOctnodes_o); //occasionally currentSize is 0...
	check(error);
	CLFW::DefaultQueue.finish();

	/* Use the internal octree nodes to calculate leaves */
	error |= GetLeaves_p(octree_o, totalOctnodes_o, leaves_o, totalLeaves_o);
	check(error);
	CLFW::DefaultQueue.finish();

	return error;
}

cl_int Quadtree::resolveAmbiguousCells(
  cl::Buffer octree_i, 
  cl_int numOctNodes, 
  cl::Buffer leaves_i, 
  cl_int numLeaves, 
  cl::Buffer lines_i, 
  cl_int totalLines, 
  cl::Buffer qpoints_i,
	cl_int totalPoints,
  cl::Buffer LCPToLine_i,
  cl::Buffer LCPBounds_i
) { 
	if (totalLines <= 1) return CL_SUCCESS;
  using namespace Kernels;
  cl::Buffer sparseConflicts;
  cl_int error = 0;

  /* Determine points to resolve conflict cells */
  error |= FindConflictCells_p( octree_i, leaves_i, numLeaves, LCPToLine_i,
		LCPBounds_i, lines_i, totalLines, qpoints_i, resln.width, sparseConflicts);
	check(error);

	//vector<Conflict>conflicts_vec;
	//CLFW::Download<Conflict>(sparseConflicts, numLeaves, conflicts_vec);
	//writeToFile<Conflict>(conflicts_vec, "TestData//simple//sparseConflicts.bin");
	//cout << endl;

	/* Remove the non-conflict cells */
	cl::Buffer cPred, cAddr, conflicts;
	cl_int numConflicts;
	error |= CLFW::get(conflicts, "conflicts", sizeof(Conflict) * numLeaves);
	error |= CLFW::get(cPred, "cPred", sizeof(cl_int) * numLeaves);
	error |= CLFW::get(cAddr, "cAddr", sizeof(cl_int) * numLeaves);
	error |= PredicateConflicts_p(sparseConflicts, numLeaves, "", cPred);
	error |= StreamScan_p(cPred, numLeaves, "cnflctaddr",  cAddr);
	error |= CLFW::Download<cl_int>(cAddr, numLeaves - 1, numConflicts);
	error |= CompactConflicts_p(sparseConflicts, cPred, cAddr, numLeaves, conflicts);

	/* Use the conflicts to generate resolution points */
	cl::Buffer conflictInfo, numPtsPerConflict, scannedNumPtsPerConflict, predPntToConflict, pntToConflict;
	cl_int numResPts;
	error |= GetResolutionPointsInfo_p(conflicts, numConflicts, qpoints, conflictInfo, numPtsPerConflict);

	/*vector<ConflictInfo> ci;
	vector<cl_int> numc;
	CLFW::Download<ConflictInfo>(conflictInfo, numConflicts, ci);
	writeToFile<ConflictInfo>(ci, "TestData//simple//conflictInfo.bin");
	CLFW::Download<cl_int>(numPtsPerConflict, numConflicts, numc);
*/


	error |= CLFW::get(scannedNumPtsPerConflict, "snptspercnflct", nextPow2(sizeof(cl_int) * numConflicts));
	error |= StreamScan_p(numPtsPerConflict, numConflicts, "conflictInfo", scannedNumPtsPerConflict);
	error |= CLFW::Download<cl_int>(scannedNumPtsPerConflict, numConflicts - 1, numResPts);
	error |= PredicatePointToConflict_p(scannedNumPtsPerConflict, numConflicts, numResPts, predPntToConflict);
	error |= CLFW::get(pntToConflict, "pnt2Conflict", nextPow2(sizeof(cl_int) * numResPts));
	error |= StreamScan_p(predPntToConflict, numResPts, "pnt2Conf",  pntToConflict);


	//vector<cl_int> pointToInfo_;
	//CLFW::Download<cl_int>(pntToConflict, numResPts, pointToInfo_);
	//writeToFile<cl_int>(pointToInfo_, "TestData//simple//pntToConflict.bin");
	//cout << endl;
//	vector<Conflict> conflicts_vec;
//	CLFW::Download<Conflict>(conflicts, numConflicts, conflicts_vec);
//	writeToFile<Conflict>(conflicts_vec, "TestData//simple//conflicts.bin");
////	writeToFile<cl_int>(numConflicts, "TestData//simple//numConflicts.bin");

	/* Determine the number of points needed to solve each conflict. */
	//cl::Buffer conflictInfoBuffer, resolutionCounts,
	//	resolutionPredicates, scannedCounts;
	//error |= GetResolutionPointsInfo_p(
	//	totalLeaves, conflicts, lines_i,
	//	qpoints_i, conflictInfoBuffer, resolutionCounts,
	//	resolutionPredicates);
	//
  //if (lines.size() > 1) {
  //  //  do {
  //  
  //  CLFW::DefaultQueue = CLFW::Queues[0];
  //  cl::Buffer resZPoints;

  //  getResolutionPoints();
  //  drawResolutionPoints();

  //  /* */
  //  //      getZOrderPoints(resQPoints, resZPoints, "rZPoints", totalResPoints);
  //  ////      addResolutionPoints();
  //  //      int otherOctreeSize;
  //  //      cl::Buffer otherOctree;
  //  //      getVertexOctree(resZPoints, totalResPoints, otherOctree, "otherOctree", otherOctreeSize);
  //  //   
  //  //    vector<OctNode> otherOctreevec1, otherOctreevec2;
  //  //    cl_int error = Kernels::DownloadOctNodes(otherOctreevec1, otherOctree, otherOctreeSize);
  //  //cl::Buffer octnodePredication, octnodeAddresses;
  //  //    // Create duplicate predication
  //  //PredicateDuplicateNodes_p(octreeBuffer, otherOctree, otherOctreeSize, octnodePredication);
  //  //    error |= CLFW::DefaultQueue.finish();
  //  //    error |= Kernels::DownloadOctNodes(otherOctreevec2, otherOctree, otherOctreeSize);
  //  //vector<cl_int> addrs;

  //  // Perform Octnode Compaction
  //  /*CLFW::get(octnodeAddresses, "octaddrs", Kernels::nextPow2(otherOctreeSize) * sizeof(cl_int));
  //  Kernels::StreamScan_p(octnodePredication, octnodeAddresses, Kernels::nextPow2(otherOctreeSize), "mergeScanI");
  //  Kernels::Download<cl_int>(octnodeAddresses, addrs, otherOctreeSize);
  //  cout << addrs[addrs.size() - 1] << endl;*/

  //  //    cl::Buffer compactNodes;
  //  //    CLFW::get(compactNodes, "compactNodes", Kernels::nextPow2(otherOctreeSize) * sizeof(OctNode));
  //  //   // Kernels::OctnodeDoubleCompact(otherOctree, compactNodes, octnodePredication, octnodeAddresses, Kernels::nextPow2(otherOctreeSize));

  //  //    // Merge octrees
  //  //    cl::Buffer mergedOctree;
  //  //    CLFW::get(mergedOctree, "mergedOctree", (octreeSize + addrs[addrs.size() - 1]) * sizeof(OctNode));
  //  ////    // Repair indices

  //  //    //cout << otherOctreeSize << endl;

  //  //    //  //getZOrderPoints(); //TODO: only z-order the additional points here...

  //  //    //  /*
  //  //    //    On one queue, sort the lines for the ambiguous cell detection.
  //  //    //    On the other, start building the karras octree.
  //  //    //  */
  //  //    //  
  //  //    //  /* Identify the cells that contain more than one object. */
  //  //    //  if (lines.size() == 0 || lines.size() == 1) break;
  //  //    //  /* Resolve one iteration of conflicts. */
  //  //    drawResolutionPoints();
  //  //    //  break;
  //  //    //  if (totalIterations == Options::maxConflictIterations) {
  //  //    //    //LOG4CPLUS_WARN(logger, "*** Warning: breaking out of conflict "
  //  //    //      //<< "detection loop early for debugging purposes.");
  //  //    //    // exit(0);
  //  //    break;
  //  //    //  }
  //  //  } while (previousSize != octreeSize && totalResPoints != 0);
  //}
  return error;
}

void Quadtree::clear() {
  using namespace GLUtilities;
  Sketcher::instance()->clear();
  initialOctreeSize = 0;
  totalResPoints = 0;

  points.clear();
  gl_instances.clear();
  octree.clear();
  resolutionPoints.clear();
}

void Quadtree::build(const PolyLines *polyLines) {
  using namespace Kernels;
  CLFW::DefaultQueue = CLFW::Queues[0];
  cl_int error = 0;

  /* Clear the old quadtree */
  clear();  

  /* Extract points from objects, and calculate a bounding box. */
  getPoints(polyLines, points, pointColors, lines);

  if (points.size() == 0) return;
  getBoundingBox(points, points.size(), bb);

  /* Upload the data to OpenCL buffers */
	error |= CLFW::get(pointsBuffer, "pts", points.size() * sizeof(floatn));
	error |= CLFW::get(pntColorsBuffer, "ptcolr", points.size() * sizeof(floatn));
  error |= CLFW::get(linesBuffer, "lines", lines.size()*sizeof(Line));
  error |= CLFW::Upload<floatn>(points, pointsBuffer);
	error |= CLFW::Upload<cl_int>(pointColors, pntColorsBuffer);
  error |= CLFW::Upload<Line>(lines, linesBuffer);
  check(error);

  /* Place the points on a Z-Order curve */
  error |= placePointsOnCurve(pointsBuffer, points.size(), resln, bb, "initial", qpoints, zpoints);
  check(error);
	
  /* On one queue, build the initial vertex octree */
  CLFW::DefaultQueue = CLFW::Queues[0];
  error |= buildPrunedOctree(zpoints, pntColorsBuffer, points.size(), resln, bb, "initial", octreeBuffer, initialOctreeSize, leavesBuffer, totalLeaves);
  check(error);
  
	/* On another queue, compute line bounding cells and generate the unordered line indices. */
  CLFW::DefaultQueue = CLFW::Queues[1];
  error |= GetLineLCPs_p(linesBuffer, lines.size(), zpoints, resln.mbits, LineLCPs);
	vector<BigUnsigned>zpoints_vec;
	CLFW::Download<BigUnsigned>(zpoints, points.size(), zpoints_vec);
	writeToFile<BigUnsigned>(zpoints_vec, "TestData//simple//zpoints.bin");
	vector<LCP> LineLCPs_vec;
	CLFW::Download<LCP>(LineLCPs, lines.size(), LineLCPs_vec);
	writeToFile(LineLCPs_vec, "TestData//simple//line_lcps.bin");
  error |= InitializeFacetIndices_p(lines.size(), lineIndices);
  check(error);

  CLFW::Queues[0].finish();
  CLFW::Queues[1].finish();

  /* For each bounding cell, look up it's surrounding octnode in the tree. */
  error |= LookUpOctnodeFromLCP_p(LineLCPs, lines.size(), octreeBuffer, LCPToOctNode);
  check(error);

  /* Sort the node to line pairs by key. This gives us a node to facet mapping for conflict cell detection. */
  error |= RadixSortPairsByKey(LCPToOctNode, lineIndices, lines.size());
  check(error);

  /* For each octnode, determine the first and last bounding cell index to be used for conflict cell detection. */
  error |= GetLCPBounds_p(LCPToOctNode, lines.size(), initialOctreeSize, LCPBounds);
  check(error);

  /* Finally, resolve the ambiguous cells. */
  error |= resolveAmbiguousCells(octreeBuffer, initialOctreeSize, leavesBuffer, totalLeaves, 
		linesBuffer, lines.size(), qpoints, points.size(), lineIndices, LCPBounds);
  check(error);

  /* Read back the octree. */
  octree.resize(initialOctreeSize);
  error |= CLFW::DefaultQueue.enqueueReadBuffer( octreeBuffer, CL_TRUE, 0, sizeof(OctNode)*initialOctreeSize, octree.data());
  check(error);

  /* Add the octnodes and conflict cells so they'll be rendered with OpenGL. */
  addOctreeNodes();
  addConflictCells();
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

//void Quadtree::getQuantizedPoints() {
//  check(Kernels::QuantizePoints_p(totalPoints, karrasPointsBuffer, quantizedPointsBuffer, bb.minimum, resln.width, bb.maxwidth));
//}
//
//void Quadtree::getZOrderPoints(cl::Buffer qPoints, cl::Buffer &zpoints, string zPointsName, int totalPoints) {
//  check(Kernels::PointsToMorton_p(qPoints, zpoints, zPointsName, totalPoints, resln.bits));
//}
//
//void Quadtree::getUnorderedBCellFacetPairs() {
//  check(Kernels::GetBCellLCP_p(linesBuffer, zpoints, BCells,
//    unorderedLineIndices, lines.size(), resln.mbits));
//}
//
//void Quadtree::getVertexOctree(cl::Buffer zpoints_i, cl_int numZPoints, cl::Buffer &octree_o, string octreeName, int &octreeSize_o) {
//  check(Kernels::BuildOctree_p(
//    zpoints_i, numZPoints, octree_o, octreeName, octreeSize_o, resln.bits, resln.mbits));
//}


/*
Quick microsecond timer:

start = std::chrono::steady_clock::now();
end = std::chrono::steady_clock::now();
elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
std::cout << "sort:" << elapsed.count() << " microseconds." << std::endl;
*/
//void Quadtree::getConflictCells(cl::Buffer octree_i) {
//
//}

//void Quadtree::getResPoints(
//	cl::Buffer conflicts_i, 
//	int numConflicts, 
//	cl::Buffer lines_i, 
//	cl::Buffer qpoints_i) 
//{
//	using namespace Kernels;
//	cl_int error = 0;
////  if (lines.size() < 2) return;
////  resolutionPoints.resize(0);
////  totalResPoints = 0;
////
//
////  check(CLFW::get(
////    scannedCounts, "sResCnts",
////    Kernels::nextPow2(totalLeaves) * sizeof(cl_int)));
////
////  check(Kernels::StreamScan_p(
////    resolutionCounts, scannedCounts, Kernels::nextPow2(totalLeaves),
////    "resolutionIntermediate"));
////
////  check(CLFW::DefaultQueue.enqueueReadBuffer(
////    scannedCounts, CL_TRUE,
////    (totalLeaves * sizeof(cl_int)) - sizeof(cl_int), sizeof(cl_int),
////    &totalResPoints));
////
////  if (totalResPoints > 100000) {
////    vector<cl_int> resolutionCountsVec;
////    vector<cl_int> scannedCountsVec;
////    Download<cl_int>(resolutionCounts, resolutionCountsVec, Kernels::nextPow2(totalLeaves));
////    Download<cl_int>(scannedCounts, scannedCountsVec, Kernels::nextPow2(totalLeaves));
////    cout << "Something likely went wrong. " << totalResPoints << " seems like a lot of resolution points.." << endl;
////    totalResPoints = 0;
////    return;
////  }
////
////  if (totalResPoints < 0) {
////    return;
////  }
////  check(Kernels::GetResolutionPoints_p(
////    totalLeaves, totalResPoints, conflictsBuffer, linesBuffer,
////    quantizedPointsBuffer, conflictInfoBuffer, scannedCounts,
////    resolutionPredicates, resQPoints));
//}
//
//void Quadtree::addResolutionPoints() {
//  benchmark("addResolutionPoints");
//  if (totalResPoints < 0) {
//    cout << "Error computing resolution points." << endl;
//    return;
//  }
//  if (totalResPoints != 0) {
//    int original = totalPoints;
//    int additional = totalResPoints;
//    totalPoints += additional;
//    cl_int error = 0;
//
//    //If the new resolution points wont fit inside the existing buffer.
//    if (original + additional > nextPow2(original)) {
//      cl::Buffer oldQPointsBuffer = quantizedPointsBuffer;
//      CLFW::Buffers["qPoints"] = cl::Buffer(CLFW::Contexts[0], CL_MEM_READ_WRITE, nextPow2(original + additional) * sizeof(intn));
//      quantizedPointsBuffer = CLFW::Buffers["qPoints"];
//      error |= CLFW::DefaultQueue.enqueueCopyBuffer(oldQPointsBuffer, quantizedPointsBuffer, 0, 0, original * sizeof(intn));
//    }
//    error |= CLFW::DefaultQueue.enqueueCopyBuffer(resQPoints, quantizedPointsBuffer, 0, original * sizeof(intn), additional * sizeof(intn));
//    assert_cl_error(error);
//  }
//  if (benchmarking) CLFW::DefaultQueue.finish();
//}

/* Drawing Methods */
void Quadtree::addOctreeNodes() {
  floatn temp;
  floatn center;

  if (initialOctreeSize == 0) return;
  center = (bb.minimum + bb.maxwidth*.5);
  float3 color = { 0.75, 0.75, 0.75 };
  addOctreeNodes(0, center, bb.maxwidth, color);
}

void Quadtree::addOctreeNodes(int index, floatn offset, float scale, float3 color)
{
  Instance i = { offset.x, offset.y, 0.0, scale, color.x, color.y, color.z };
  gl_instances.push_back(i);
  if (index != -1) {
    OctNode current = octree[index];
    scale /= 2.0;
    float shift = scale / 2.0;
    addOctreeNodes(current.children[0], { offset.x - shift, offset.y - shift },
      scale, color);
    addOctreeNodes(current.children[1], { offset.x + shift, offset.y - shift },
      scale, color);
    addOctreeNodes(current.children[2], { offset.x - shift, offset.y + shift },
      scale, color);
    addOctreeNodes(current.children[3], { offset.x + shift, offset.y + shift },
      scale, color);
  }
}

void Quadtree::addLeaf(int internalIndex, int childIndex, float3 color) {
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

void Quadtree::addConflictCells() {
  // if (Options::debug) {
  CLFW::Download<Conflict>(CLFW::Buffers["sparseConflicts"], totalLeaves, conflicts);
  CLFW::Download<Leaf>(leavesBuffer, totalLeaves, leaves);
  if (conflicts.size() == 0) return;
  for (int i = 0; i < conflicts.size(); ++i) {
    if (conflicts[i].color == -2)
    {
      addLeaf(leaves[i].parent, leaves[i].quadrant,
      { Options::conflict_color[0],
        Options::conflict_color[1],
        Options::conflict_color[2] });
    }
  }
}

void Quadtree::drawResolutionPoints() {
  using namespace GLUtilities;
  vector<intn> resolutionPoints;
  CLFW::Download<intn>(resQPoints, totalResPoints, resolutionPoints);
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