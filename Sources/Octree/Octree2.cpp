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
	CLFW::getBuffer(zpoints_copy, uniqueString + "zptscpy",
		nextPow2(sizeof(big) * totalPoints));
	error |= CLFW::DefaultQueue.enqueueCopyBuffer(zpoints_i, zpoints_copy,
		0, 0, totalPoints * sizeof(big));
	check(error);

	/* Radix sort the zpoints */
	error |= RadixSortBig_p(zpoints_copy, totalPoints,
		resln.mbits, uniqueString);
	check(error);

	/* Unique the zpoints */
	error |= UniqueSorted(zpoints_copy, totalPoints, uniqueString,
		uniqueTotalPoints);
	check(error);

	/* Build a binary radix tree*/
	error |= BuildBinaryRadixTree_p(zpoints_copy, uniqueTotalPoints,
		resln.mbits, uniqueString, brt);
	check(error);

	vector<cl_int> nullvec;
	vector<BrtNode> brt_vec;
	vector<OctNode> octree_vec;
	CLFW::Download<BrtNode>(brt, uniqueTotalPoints - 1, brt_vec);
	BinaryRadixToOctree_s(brt_vec, false, nullvec, octree_vec);


	/* Convert the binary radix tree to an octree*/
	error |= BinaryRadixToOctree_p(brt, false, nullBuffer, uniqueTotalPoints,
		uniqueString, octree_o, totalOctnodes_o); //occasionally currentSize is 0...
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
	CLFW::getBuffer(zpoints_copy, uniqueString + "zptscpy", sizeof(big) * nextPow2(totalPoints));
	error |= CLFW::DefaultQueue.enqueueCopyBuffer(zpoints_i, zpoints_copy,
		0, 0, totalPoints * sizeof(big));

	CLFW::getBuffer(pntColors_copy, uniqueString + "colscpy", sizeof(cl_int) * nextPow2(totalPoints));
	error |= CLFW::DefaultQueue.enqueueCopyBuffer(pntColors_i, pntColors_copy,
		0, 0, sizeof(cl_int) * totalPoints);
	
	/* Radix sort the zpoints */
	error |= RadixSortBigToInt_p(zpoints_copy, pntColors_copy, totalPoints, resln.mbits, "sortZpoints");
	check(error);
	
	/* Unique the zpoints */
	error |= UniqueSortedBUIntPair(zpoints_copy, pntColors_copy, totalPoints, uniqueString,
		uniqueTotalPoints);
	check(error);

	/* Build a colored binary radix tree*/
	error |= BuildColoredBinaryRadixTree_p(zpoints_copy, pntColors_copy, uniqueTotalPoints,
		resln.mbits, uniqueString, brt, brtColors);
	check(error);

	/* Identify required cells */
	error |= PropagateBRTColors_p(brt, brtColors, uniqueTotalPoints - 1, uniqueString);
	check(error);
	
	/* Convert the binary radix tree to an octree*/
	error |= BinaryRadixToOctree_p(brt, true, brtColors, uniqueTotalPoints, uniqueString,
		octree_o, totalOctnodes_o); //occasionally currentSize is 0...
	check(error);

	/* Use the internal octree nodes to calculate leaves */
	error |= GetLeaves_p(octree_o, totalOctnodes_o, leaves_o, totalLeaves_o);
	check(error);

	return error;
}

cl_int Quadtree::initializeConflictCellDetection(
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

	/* Sort the node to line pairs by key. This gives us a node to facet mapping
	for conflict cell detection. */
	error |= RadixSortIntToInt_p(LCPToOctNode, lineIndices_o, numLines, 8 * sizeof(cl_int), "facetMapping");
	check(error);

	/* For each octnode, determine the first and last bounding cell index to be
	used for conflict cell detection. */
	error |= GetLCPBounds_p(LCPToOctNode, numLines, numOctNodes, LCPBounds_o);
	check(error);
	return error;
}

cl_int Quadtree::findConflictCells(
	cl::Buffer &octree_i,
	cl_int numOctNodes,
	cl::Buffer &leaves_i,
	cl_int numLeaves,
	cl::Buffer &qpoints_i,
	cl::Buffer &zpoints_i,
	cl::Buffer &lines_i,
	cl_int numLines,
	bool keepCollisions,
	Resln &resln,
	cl::Buffer &conflicts_o,
	cl_int &numConflicts
) {
	using namespace Kernels;
	cl_int error = 0;

	/* Initialize the node to facet mapping*/
	cl::Buffer LineLCPs, lineIndices, LCPBounds;
	error |= initializeConflictCellDetection(zpoints_i, lines_i, numLines, resln,
		octree_i, numOctNodes, lineIndices, LCPBounds);

	/* Use that mapping to find conflict cells*/
	cl::Buffer sparseConflicts;
	error |= FindConflictCells_p(octree_i, leaves_i, numLeaves, lineIndices,
		LCPBounds, lines_i, numLines, qpoints_i, resln.width, keepCollisions, sparseConflicts);
	check(error);
	
	/* Compact the non-conflict cells to the right */
	cl::Buffer cPred, cAddr;
	error |= CLFW::getBuffer(cPred, "cPred", sizeof(cl_int) * nextPow2(numLeaves));
	error |= CLFW::getBuffer(cAddr, "cAddr", sizeof(cl_int) * nextPow2(numLeaves));
	error |= PredicateConflicts_p(sparseConflicts, numLeaves, "", cPred);
	error |= StreamScan_p(cPred, numLeaves, "cnflctaddr", cAddr);
	error |= CLFW::Download<cl_int>(cAddr, numLeaves - 1, numConflicts);
	error |= CLFW::getBuffer(conflicts_o, "conflicts", sizeof(Conflict) * nextPow2(numLeaves));
	error |= CompactConflicts_p(sparseConflicts, cPred, cAddr, numLeaves, conflicts_o);
	check(error);
	return error;
}

cl_int Quadtree::generateResolutionPoints(
	cl::Buffer &conflicts_i,
	cl_int numConflicts,
	Resln &resln,
	cl::Buffer &qpoints_i,
	cl::Buffer &resPts,
	cl::Buffer &resZPts,
	cl_int &numResPts
) {
	using namespace Kernels;
	cl_int error = 0;

	/* Use the conflicts to initialize data required to calculate resolution points */
	cl::Buffer conflictInfo, numPtsPerConflict, scannedNumPtsPerConflict, predPntToConflict,
		pntToConflict;
	error |= GetResolutionPointsInfo_p(conflicts_i, numConflicts, qpoints_i, conflictInfo,
		numPtsPerConflict);
	/* Scan the pts per conflict to determine beginning and ending addresses for res pts*/
	error |= CLFW::getBuffer(scannedNumPtsPerConflict, "snptspercnflct",
		sizeof(cl_int) * nextPow2(numConflicts));
	error |= StreamScan_p(numPtsPerConflict, numConflicts, "conflictInfo",
		scannedNumPtsPerConflict);
	error |= CLFW::Download<cl_int>(scannedNumPtsPerConflict, numConflicts - 1, numResPts);
	/* Create a res pnt to conflict info mapping so we can determine resolution points
	in parallel. */
	error |= PredicatePointToConflict_p(scannedNumPtsPerConflict, numConflicts, numResPts,
		predPntToConflict);
	error |= CLFW::getBuffer(pntToConflict, "pnt2Conflict", nextPow2(sizeof(cl_int) * numResPts));
	error |= StreamScan_p(predPntToConflict, numResPts, "pnt2Conf", pntToConflict);
	check(error);

	/* Get the resolution points */
	error |= GetResolutionPoints_p(conflicts_i, conflictInfo, scannedNumPtsPerConflict,
		numResPts, pntToConflict, qpoints_i, resPts);

	if (Options::showResolutionPoints) {
		int offset = resolutionPoints.size();
		resolutionPoints.resize(resolutionPoints.size() + numResPts);
		vector<intn> temp;
		CLFW::Download<intn>(resPts, numResPts, temp);
		for (int i = 0; i < temp.size(); ++i) {
			resolutionPoints[offset + i] = UnquantizePoint(&temp[i], &bb.minimum, resln.width, bb.maxwidth);
		}
	}

	/* Convert to z-order */
	error |= QPointsToZPoints_p(resPts, numResPts, resln.bits, "res", resZPts);
	check(error);
	return error;
}

cl_int Quadtree::combinePoints(
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
	error |= CLFW::getBuffer(combinedQPts_o, "qpoints" + std::to_string(iteration),
		nextPow2(numPts + numResPts) * sizeof(intn));
	error |= queue.enqueueCopyBuffer(qpoints_i, combinedQPts_o, 0, 0, numPts * sizeof(intn));
	error |= queue.enqueueCopyBuffer(resPts_i, combinedQPts_o, 0,
		numPts * sizeof(intn), numResPts * sizeof(intn));
	check(error);

	error |= CLFW::getBuffer(combinedZPts_o, "zpoints" + std::to_string(iteration),
		nextPow2(numPts + numResPts) * sizeof(big));
	error |= queue.enqueueCopyBuffer(zpoints_i, combinedZPts_o, 0, 0, numPts * sizeof(big));
	error |= queue.enqueueCopyBuffer(resZPts_i, combinedZPts_o, 0, numPts * sizeof(big),
		numResPts * sizeof(big));
	check(error);

	error |= CLFW::getBuffer(combinedCols_o, "ptcol" + std::to_string(iteration),
		nextPow2(numPts + numResPts) * sizeof(cl_int));
	error |= queue.enqueueCopyBuffer(pntCols_i, combinedCols_o, 0, 0, numPts * sizeof(cl_int));
	vector<cl_int> resCols(numResPts);
	for (int i = 0; i < numResPts; ++i) resCols[i] = -3 - i;
	error |= queue.enqueueWriteBuffer(combinedCols_o, CL_TRUE, numPts * sizeof(cl_int),
		numResPts * sizeof(cl_int), resCols.data());
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
	using namespace Kernels;
	if (numLines <= 1) return CL_SUCCESS;
	cl_int error = 0;
	if (iteration >= Options::maxConflictIterations) {
		if (Options::showObjectIntersections) {
			error |= findConflictCells(octree_i, numOctNodes, leaves_i, numLeaves,
				qpoints_i, zpoints_i, lines_i, numLines, true, resln, conflictsBuffer, numConflicts);
			CLFW::Download<Conflict>(conflictsBuffer, numConflicts, conflicts);
		}
		return error;
	};

	/* Determine conflicts to resolve */
	error |= findConflictCells(octree_i, numOctNodes, leaves_i, numLeaves,
		qpoints_i, zpoints_i, lines_i, numLines, false, resln, conflictsBuffer, numConflicts);
	check(error);

	if (numConflicts == 0) return error;

	/* Use the conflicts to generate resolution points */
	cl::Buffer resPts, resZPoints; cl_int numResPts;
	error |= generateResolutionPoints(conflictsBuffer, numConflicts, resln, qpoints_i, resPts, resZPoints, numResPts);
	check(error);

	/* Combine the original and generated resolution points */
	cl::Buffer combinedQPts, combinedZPts, combinedCols;
	combinePoints(qpoints_i, zpoints_i, pntCols_i, numPts,
		resPts, resZPoints, numResPts, iteration, combinedQPts, combinedZPts, combinedCols);
	check(error);


	/* Build an octree from the combined points */
	cl::Buffer combinedOctree, combinedLeaves;
	cl_int combinedOctSize, combinedLeafSize;
	if (Options::pruneOctree)
		error |= buildPrunedOctree(combinedZPts, combinedCols, numPts + numResPts, resln, bb,
			"res" + iteration, combinedOctree, combinedOctSize, combinedLeaves, combinedLeafSize);
	else
		error |= buildVertexOctree(combinedZPts, numPts + numResPts, resln, bb,
			"res" + iteration, combinedOctree, combinedOctSize, combinedLeaves, combinedLeafSize);
	check(error);
	int originalOctSize = numOctNodes;
	octree_i = combinedOctree;
	numOctNodes = combinedOctSize;
	leaves_i = combinedLeaves;
	numLeaves = combinedLeafSize;

	/* If the resolution points don't effect the octree, quit resolving. */
	if (combinedOctSize == originalOctSize) {
		if (Options::showObjectIntersections) {
			error |= findConflictCells(octree_i, numOctNodes, leaves_i, numLeaves,
				qpoints_i, zpoints_i, lines_i, numLines, true, resln, conflictsBuffer, numConflicts);
			CLFW::Download<Conflict>(conflictsBuffer, numConflicts, conflicts);
		}
		return error;
	}

	/* resolve further conflicts */
	if (iteration < Options::maxConflictIterations)
		resolveAmbiguousCells(octree_i, numOctNodes, leaves_i, numLeaves,
			lines_i, numLines, combinedQPts,
			combinedZPts, combinedCols, numPts + numResPts, iteration + 1);
	else {
		if (Options::showObjectIntersections) {
			error |= findConflictCells(octree_i, numOctNodes, leaves_i, numLeaves,
				qpoints_i, zpoints_i, lines_i, numLines, true, resln, conflictsBuffer, numConflicts);
			CLFW::Download<Conflict>(conflictsBuffer, numConflicts, conflicts);
		}
	}

	return error;
}

void Quadtree::clear() {
	octreeSize = 0;
	totalResPoints = 0;
	nodes.clear();
	conflicts.clear();
	resolutionPoints.clear();
}

void Quadtree::build_internal (cl_int numPts, cl_int numLines) {
	using namespace Kernels;
	CLFW::DefaultQueue = CLFW::Queues[0];
	cl_int error = 0;

	/* Place the points on a Z-Order curve */
	error |= placePointsOnCurve(pointsBuffer, numPts, resln, bb, "initial", qpoints, zpoints);
	check(error);

	/* Build the initial octree */
	CLFW::DefaultQueue = CLFW::Queues[0];
	if (Options::pruneOctree)
		error |= buildPrunedOctree(zpoints, pntColorsBuffer, numPts, resln, bb, "initial",
			octreeBuffer, octreeSize, leavesBuffer, totalLeaves);
	else
		error |= buildVertexOctree(zpoints, numPts, resln, bb, "initial", octreeBuffer,
			octreeSize, leavesBuffer, totalLeaves);
	check(error);

	/* Finally, resolve the ambiguous cells. */
	if (resolutionRequired) {
		error |= resolveAmbiguousCells(octreeBuffer, octreeSize, leavesBuffer, totalLeaves,
			linesBuffer, numLines, qpoints, zpoints, pntColorsBuffer, numPts, 0);
		check(error);
	}

	/* Download the octree for CPU usage */
	error |= CLFW::Download<OctNode>(octreeBuffer, octreeSize, nodes);
	check(error);
}

void Quadtree::build(vector<floatn> &points, vector<cl_int> &pointColors, vector<Line> &lines,
	BoundingBox bb) {
	using namespace Kernels;
	CLFW::DefaultQueue = CLFW::Queues[0];
	cl_int error = 0;

	/* Clear the old quadtree */
	clear();

	if (points.size() == 0) return;

	this->bb = bb;

	/* Upload the data to OpenCL buffers */
	error |= CLFW::getBuffer(pointsBuffer, "pts", CLFW::NextPow2(points.size()) * sizeof(floatn));
	error |= CLFW::getBuffer(pntColorsBuffer, "ptcolr", CLFW::NextPow2(points.size()) * sizeof(cl_int));
	error |= CLFW::getBuffer(linesBuffer, "lines", CLFW::NextPow2(lines.size()) * sizeof(Line));
	check(error);

	error |= CLFW::Upload<floatn>(points, pointsBuffer);
	error |= CLFW::Upload<cl_int>(pointColors, pntColorsBuffer);
	error |= CLFW::Upload<Line>(lines, linesBuffer);
	check(error);

	//glm::mat4 matrix(1.0);
	////matrix = glm::translate(matrix, glm::vec3(.2, .2, 0.0));
	//error |= multiplyM4V4_p(untranslatedPointsBuffer, points.size(), matrix, "translated", pointsBuffer);
	//CLFW::DefaultQueue.finish();
	//check(error);

	build_internal(points.size(), lines.size());
}

void Quadtree::build(const PolyLines *polyLines) {
	using namespace Kernels;
	CLFW::DefaultQueue = CLFW::Queues[0];
	cl_int error = 0;

	/* Clear the old quadtree */
	clear();

	/* Don't build a quadtree if there are no points. */
	if (polyLines->points.size() == 0) return;

	//TODO: parallelize this calculation...
	getBoundingBox(polyLines->points, polyLines->points.size(), bb);
	resolutionRequired = polyLines->lasts.size() > 1;

	this->pointsBuffer = polyLines->pointsBuffer;
	this->pntColorsBuffer = polyLines->pointColorsBuffer;
	this->linesBuffer = polyLines->linesBuffer;

	build_internal(polyLines->points.size(), polyLines->lines.size());
}

void Quadtree::build(const Polygons* polygons) {
	using namespace Kernels;
	CLFW::DefaultQueue = CLFW::Queues[0];
	cl_int error = 0;

	/* Clear the old quadtree */
	clear();

	/* Don't build a quadtree if there are no points. */
	if (polygons->points.size() == 0) return;

	//TODO: parallelize this calculation...
	getBoundingBox(polygons->points, polygons->points.size(), bb);
	resolutionRequired = polygons->lasts.size() > 1;
	//bb.maxwidth = 4.0;
	//bb.minimum = make_float2(-2.4, -1.6);
	//bb.maximum = make_float2(2.4, 1.6);

	this->pointsBuffer = polygons->pointsBuffer;
	this->pntColorsBuffer = polygons->pointColorsBuffer;
	this->linesBuffer = polygons->linesBuffer;

	build_internal(polygons->points.size(), polygons->lines.size());
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

//void Quadtree::getPoints(const PolyLines *polyLines, vector<floatn> &points,
//	vector<cl_int> &pointColors, std::vector<Line> &lines) {
//	benchmark("getPoints");
//
//	const vector<vector<floatn>> polygons = polyLines->getPolygons();
//	lines = polyLines->getLines();
//	
//	resolutionRequired = (polygons.size() > 1);
//
//	// Get all vertices into a 1D array.
//	for (int i = 0; i < polygons.size(); ++i) {
//		const vector<floatn>& polygon = polygons[i];
//		for (int j = 0; j < polygon.size(); ++j) {
//			points.push_back(polygon[j]);
//			pointColors.push_back(i);
//		}
//	}
//}

void Quadtree::getBoundingBox(const vector<floatn> &points, const int totalPoints,
	BoundingBox &bb) {
	benchmark("getBoundingBox");

	if (Options::bbxmin == -1 && Options::bbxmax == -1) {
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
		bb.minimum = make_floatn(Options::bbxmin, Options::bbymin);
		bb.maximum = make_floatn(Options::bbxmax, Options::bbymax);
		bb.maxwidth = BB_max_size(&bb);
	}
}

#undef benchmark
#undef check