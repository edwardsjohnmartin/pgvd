#pragma once
#ifdef __APPLE__
#include "OpenCL/cl.hpp"
#else
#include <CL/cl.hpp>
#endif
#include "clfw.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include "timer.h"

extern "C" {
  #include "z_order.h"
  #include "BrtNode.h"
  #include "BuildBRT.h"
  #include "OctNode.h"
  #include "BuildOctree.h"
  #include "ParallelAlgorithms.h"
  #include "./Resln.h"
}

using namespace std;
namespace Kernels {
  void startBenchmark(string benchmarkName);
  void stopBenchmark();

  int nextPow2(int num);
  cl_int UploadPoints(const vector<intn> &points, cl::Buffer &pointsBuffer);
  cl_int PointsToMorton_p(cl::Buffer &points, cl::Buffer &zpoints, cl_int size, cl_int bits);
  cl_int PointsToMorton_s(cl_int size, cl_int bits, cl_int2* points, BigUnsigned* result);
  cl_int BitPredicate(cl::Buffer &input, cl::Buffer &predicate, unsigned int &index, unsigned char compared, cl_int globalSize);
  cl_int UniquePredicate(cl::Buffer &input, cl::Buffer &predicate, cl_int globalSize);
  cl_int StreamScan_p(cl::Buffer &input, cl::Buffer &result, cl_int globalSize);
  cl_int StreamScan_s(unsigned int* buffer, unsigned int* result, const int size);
  cl_int SingleCompact(cl::Buffer &input, cl::Buffer &result, cl::Memory &predicate, cl::Buffer &address, cl_int globalSize);
  cl_int DoubleCompact(cl::Buffer &input, cl::Buffer &result, cl::Buffer &predicate, cl::Buffer &address, cl_int globalSize);
  cl_int UniqueSorted(cl::Buffer &input, cl_int &size);
  cl_int RadixSortBigUnsigned(cl::Buffer &input, cl_int size, cl_int mbits);
  cl_int BuildBinaryRadixTree_p(cl::Buffer &zpoints, cl::Buffer &internalBRTNodes, cl_int size, cl_int mbits);
  cl_int BuildBinaryRadixTree_s(BigUnsigned* zpoints, BrtNode* internalBRTNodes, cl_int size, cl_int mbits);
  cl_int ComputeLocalSplits_p(cl::Buffer &internalBRTNodes, cl::Buffer &localSplits, cl_int size);
  cl_int ComputeLocalSplits_s(vector<BrtNode> &I, vector<unsigned int> &local_splits, const cl_int size);
  cl_int InitOctree(cl::Buffer &internalBRTNodes, cl::Buffer &octree, cl::Buffer &localSplits, cl::Buffer &scannedSplits, cl_int size, cl_int octreeSize);
  cl_int BinaryRadixToOctree_p(cl::Buffer &internalBRTNodes, vector<OctNode> &octree_vec, cl_int size);
  cl_int BinaryRadixToOctree_s(vector<BrtNode> &internalBRTNodes, vector<OctNode> &octree, cl_int size);
  cl_int BuildOctree_s(const vector<intn>& points, vector<OctNode> &octree, int bits, int mbits);
  cl_int BuildOctree_p(const vector<intn>& points, vector<OctNode> &octree, int bits, int mbits);
}