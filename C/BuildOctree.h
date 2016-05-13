#ifndef __BUILD_OCTREE_H__
#define __BUILD_OCTREE_H__

  #ifndef __OPENCL_VERSION__ 
  #include "BuildBRT.h"
  #include "../Resln.h"
  #include "../OctNode.h"
  #include "../C/BrtNode.h"
  #endif

  #ifndef __OPENCL_VERSION__
  #define __local
  #define __global
  #endif

  int quadrantInLcp(const BrtNode* brt_node, const int i);
  void ComputeLocalSplits_SerialKernel(__global unsigned int* local_splits, __global BrtNode* I, const int size);
  void ComputeLocalSplits(__global unsigned int* local_splits, __global BrtNode* I, const int gid );

  void brt2octree_init( const int brt_i, __global OctNode* octree );
  void brt2octree( const int brt_i, __global BrtNode* I, __global OctNode* octree, __global unsigned int* local_splits, __global unsigned int* prefix_sums, const int n, const int octree_size);
  void brt2octree_kernel(__global BrtNode* I, __global OctNode* octree, __global unsigned int* local_splits, __global unsigned int* prefix_sums, const int n);

#endif
