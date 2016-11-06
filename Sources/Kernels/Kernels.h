#pragma once
#ifdef __APPLE__
// #include "OpenCL/cl.hpp"
#include "./cl.hpp"
#else
#include <CL/cl.hpp>
#endif
#include "../GLUtilities/gl_utils.h"
#include "clfw.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include "../Timer/timer.h"
#include "../../SharedSources/Vector/vec.h"

#include "./glm/gtc/matrix_transform.hpp"
#include "../../SharedSources/ZOrder/z_order.h"
#include "../../SharedSources/CellResolution/Conflict.h"
#include "../../SharedSources/Quantize/Quantize.h"
extern "C" {
#include "../../SharedSources/BinaryRadixTree/BrtNode.h"
#include "../../SharedSources/BinaryRadixTree/BuildBRT.h"  
#include "../../SharedSources/ParallelAlgorithms/ParallelAlgorithms.h"  
#include "../../SharedSources/OctreeResolution/Resln.h"
#include "../../SharedSources/Line/Line.h"
}
#include "../../SharedSources/Octree/BuildOctree.h"  
#include "../../SharedSources/Octree/OctNode.h"  
#include "../../SharedSources/CellResolution/ConflictCellDetection.h"

#include "../../Sources/GLUtilities/Sketcher.h"

using namespace std;
namespace Kernels {
    void startBenchmark(string benchmarkName);
    void stopBenchmark();

    int nextPow2(int num);
    cl_int UploadKarrasPoints(const vector<floatn> &points, cl::Buffer &karrasPointsBuffer);
    cl_int UploadQuantizedPoints(const vector<intn> &points, cl::Buffer &pointsBuffer);
    cl_int UploadLines(const vector<Line> &lines, cl::Buffer &linesBuffer);
    
    cl_int DownloadInts(cl::Buffer &integersBuffer, vector<int> &integers, cl_int size);
    cl_int DownloadLines(cl::Buffer &linesBuffer, vector<Line> &lines, cl_int size);
    cl_int DownloadBoundingBoxes(cl::Buffer &boundingBoxesBuffer, vector<int> &boundingBoxes, cl_int size);
    cl_int DownloadConflicts(vector<Conflict> &conflictPairsVec, cl::Buffer &conflictPairsBuffer, cl_int size);
    cl_int DownloadBCells(vector<BCell> &BCellsVec, cl::Buffer &BCellsBuffer, cl_int size);
    cl_int DownloadFacetPairs(vector<FacetPair> &facetPairsVec, cl::Buffer &facetPairsBuffer, cl_int size);
    cl_int DownloadQPoints(vector<intn> &points, cl::Buffer &pointsBuffer, cl_int size);
    cl_int DownloadFloatnPoints(vector<floatn> &points, cl::Buffer &pointsBuffer, cl_int size);
    cl_int DownloadZPoints(vector<BigUnsigned> &zpoints, cl::Buffer &zpointsBuffer, cl_int size);

    cl_int QuantizePoints_p(cl_uint numPoints, cl::Buffer &unquantizedPoints, cl::Buffer &quantizedPoints, const floatn minimum, const int reslnWidth, const float bbWidth);
    cl_int PointsToMorton_p(cl::Buffer &points, cl::Buffer &zpoints, cl_int size, cl_int bits);
    cl_int PointsToMorton_s(cl_int size, cl_int bits, intn* points, BigUnsigned* result);
    cl_int BUBitPredicate(cl::Buffer &input, cl::Buffer &predicate, unsigned int &index, unsigned char compared, cl_int globalSize);
    cl_int GetTwoBitMask_p(cl::Buffer &input, cl::Buffer &masks, unsigned int index, unsigned char compared, cl_int size);
    cl_int GetTwoBitMask_s(BigUnsigned* input, unsigned int *masks, unsigned int index, unsigned char compared, cl_int size);
    cl_int GetFourWayPrefixSum_p(cl::Buffer &input, cl::Buffer &fourWayPrefix, unsigned int index, unsigned char compared, cl_int size);
    cl_int GetFourWayPrefixSum_s(BigUnsigned* input, unsigned int *fourWayPrefix, unsigned int index, unsigned char compared, cl_int size);
    cl_int UniquePredicate(cl::Buffer &input, cl::Buffer &predicate, cl_int globalSize);
    cl_int StreamScan_p(cl::Buffer &input, cl::Buffer &result, cl_int globalSize, string intermediateName, bool exclusive = true);
    cl_int StreamScan_s(unsigned int* buffer, unsigned int* result, const int size);
    cl_int BUSingleCompact(cl::Buffer &input, cl::Buffer &result, cl::Memory &predicate, cl::Buffer &address, cl_int globalSize);
    cl_int BUDoubleCompact(cl::Buffer &input, cl::Buffer &result, cl::Buffer &predicate, cl::Buffer &address, cl_int globalSize);
    cl_int BCellFacetDoubleCompact(cl::Buffer &inputBCells, cl::Buffer &inputIndices, cl::Buffer &resultBCells, cl::Buffer &resultIndices, cl::Buffer &predicate, cl::Buffer &address, cl_int globalSize);
    cl_int UniqueSorted(cl::Buffer &input, cl_int &size);
    cl_int RadixSortBigUnsigned_p(cl::Buffer &input, cl::Buffer &result, cl_int size, cl_int mbits);
    cl_int RadixSortPairsByKey( cl::Buffer &unsortedKeys_i, cl::Buffer &unsortedValues_i, cl::Buffer &sortedKeys_o, cl::Buffer &sortedValues_o, cl_int size);
    cl_int BuildBinaryRadixTree_p(cl::Buffer &zpoints, cl::Buffer &internalBRTNodes, cl_int size, cl_int mbits);
    cl_int BuildBinaryRadixTree_s(BigUnsigned* zpoints, BrtNode* internalBRTNodes, cl_int size, cl_int mbits);
    cl_int ComputeLocalSplits_p(cl::Buffer &internalBRTNodes, cl::Buffer &localSplits, cl_int size);
    cl_int ComputeLocalSplits_s(vector<BrtNode> &I, vector<unsigned int> &local_splits, const cl_int size);
    cl_int InitOctree(cl::Buffer &internalBRTNodes, cl::Buffer &octree, cl::Buffer &localSplits, cl::Buffer &scannedSplits, cl_int size, cl_int octreeSize);
    cl_int BinaryRadixToOctree_p(cl::Buffer &internalBRTNodes, vector<OctNode> &octree_vec, cl_int size);
    cl_int BinaryRadixToOctree_s(vector<BrtNode> &internalBRTNodes, vector<OctNode> &octree, cl_int size);
    cl_int BuildOctree_s(const vector<intn>& points, vector<OctNode> &octree, int bits, int mbits);
    cl_int BuildOctree_p(cl::Buffer zpoints, cl_int numZPoints, vector<OctNode> &octree, int bits, int mbits);
    cl_int AddAll(cl::Buffer &numbers, cl_uint& gpuSum, cl_int size);
    cl_int CheckOrder(cl::Buffer &numbers, cl_uint& gpuSum, cl_int size);
    cl_int GetBCellLCP_p(cl::Buffer &linesBuffer, cl::Buffer &zpoints, cl::Buffer &bCells, cl::Buffer &facetIndices, cl_int size, int mbits);
    cl_int GetBCellLCP_s( Line* lines, BigUnsigned* zpoints, BCell* bCells, cl_int* facetIndices, cl_int size, int mbits);
    cl_int OrderFacetsByNode_p( cl_int totalLines, cl::Buffer &lines_i, cl::Buffer &zpoints_i, cl::Buffer &NodeToBCell_o, cl::Buffer &NodeToFacet_o, const Resln &resln_i);
    cl_int LookUpOctnodeFromBCell_s(BCell* bCells, OctNode *octree, int* BCellToOctree, cl_int numBCells);
    cl_int LookUpOctnodeFromBCell_p(cl::Buffer &bCells_i, cl::Buffer &octree_i, cl::Buffer &BCellToOctnode, cl_int numBCells);
    cl_int GetFacetPairs_s(cl_int* orderedNodeIndices, FacetPair *facetPairs, cl_int numLines);
    cl_int GetFacetPairs_p(cl::Buffer &orderedNodeIndices_i, cl::Buffer &facetPairs_o, cl_int numLines, cl_int octreeSize);
    cl_int GetBCellLCP_s( Line* lines, BigUnsigned* zpoints, BCell* bCells, cl_int* facetIndices, cl_int size, int mbits);
    cl_int GetBCellLCP_p( cl::Buffer &linesBuffer, cl::Buffer &zpoints, cl::Buffer &bCells, cl::Buffer &facetIndices, cl_int size, int mbits);
    cl_int FindConflictCells_s(OctNode *octree, FacetPair *facetPairs, OctreeData *od, Conflict *conflicts, int* nodeToFacet, Line *lines, cl_int numLines, intn* points);
    cl_int FindConflictCells_p(cl::Buffer &octree, cl::Buffer &facetPairs, OctreeData &od, cl::Buffer &conflicts, cl::Buffer &nodeToFacet, cl::Buffer &lines, cl_int numLines, cl::Buffer &points);
    cl_int SampleConflictCounts_s(unsigned int totalOctnodes, Conflict *conflicts, int *totalAdditionalPoints,
        vector<int> &counts, Line* orderedlines, intn* quantizedPoints, vector<intn> &newPoints);
    cl_int GetResolutionPointsInfo_p(unsigned int totalOctnodes, cl::Buffer &conflicts, cl::Buffer &orderedLines, 
        cl::Buffer &quantizedPoints, cl::Buffer &conflictInfoBuffer, cl::Buffer &resolutionCounts, cl::Buffer &predicates);
    cl_int GetResolutionPoints_p(unsigned int totalOctnodes, unsigned int totalAdditionalPoints, cl::Buffer &conflicts,
        cl::Buffer &orderedLines, cl::Buffer &quantizedPoints, cl::Buffer &conflictInfoBuffer,
        cl::Buffer &scannedCounts, cl::Buffer &predicates, cl::Buffer &resolutionPoints);

    std::string buToString(BigUnsigned bu);
}
