#pragma once

#ifdef __OPENCL_VERSION__ 
  #include "./SharedSources/BinaryRadixTree/BuildBRT.h"
  #include "./SharedSources/Octree/OctNode.h"
  #include "./SharedSources/BinaryRadixTree/BrtNode.h"
#else
extern "C" {
  #include "../BinaryRadixTree/BuildBRT.h"
  #include "../BinaryRadixTree/BrtNode.h"
}
  #include "../Octree/OctNode.h"
  #define __local
  #define __global
#endif

int quadrantInLcp(BrtNode* brt_node, const int i);
void ComputeLocalSplits_SerialKernel(__global unsigned int* local_splits, __global BrtNode* I, const int size);
void ComputeLocalSplits(__global unsigned int* local_splits, __global BrtNode* I, const int gid );

void brt2octree_init( const int brt_i, __global OctNode* octree );
void brt2octree( const int brt_i, __global BrtNode* I, __global volatile OctNode* octree, __global unsigned int* local_splits, __global unsigned int* prefix_sums, const int n, const int octree_size);
void brt2octree_kernel(__global BrtNode* I, __global OctNode* octree, __global unsigned int* local_splits, __global unsigned int* prefix_sums, const int n);
void ComputeLeaves(__global OctNode *octree, __global Leaf *leaves, __global cl_int *leafPredicates, cl_int octreeSize, cl_int gid);
void LeafDoubleCompact(__global Leaf *inputBuffer, __global Leaf *resultBuffer, __global unsigned int *lPredicateBuffer, __global unsigned int *leftBuffer, const cl_int size, const int gid);

#ifndef __OPENCL_VERSION__
  #undef __local
  #undef __global
#endif
