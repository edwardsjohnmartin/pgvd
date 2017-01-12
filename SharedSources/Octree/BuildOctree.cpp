#ifdef  __OPENCL_VERSION__ 
  #include "./SharedSources/Octree/BuildOctree.h"
  #include "./SharedSources/ParallelAlgorithms/ParallelAlgorithms.h"
#else
  #include <stdbool.h>
  #include <stdio.h>
  #include <math.h>
  #include "BuildOctree.h"
  #include "../ParallelAlgorithms/ParallelAlgorithms.h"
  #define __local
  #define __global
#endif

void ComputeLocalSplits(__global unsigned int* local_splits, __global BrtNode* I, const int gid) {
  const int _local = I[gid].lcp_length / DIM;
  const int left = I[gid].left;
  const int right = left+1;
  if (!I[gid].left_leaf) {
    local_splits[left] = I[left].lcp_length/DIM - _local;
  }
  if (!I[gid].right_leaf) {
    local_splits[right] = I[right].lcp_length/DIM - _local;
  }
}

// local_splits stores how many times the octree needs to be
// split from a parent to a child in the brt. For example, in 2D if a child
// has an lcp_length of 8 and the parent has lcp_length of 4, then
// the child represents two octree splits.
void ComputeLocalSplits_SerialKernel(__global unsigned int* local_splits, __global BrtNode* I, const int size) {
  if (size > 0) {
    local_splits[0] = 1 + I[0].lcp_length / DIM;
  }
  for (int i = 0; i < size-1; ++i) {
    ComputeLocalSplits(local_splits, I, i);
  }
}

// Given a lcp, returns the i'th quadrant starting from the most local.
// Suppose node.lcp is 1010011 and DIM == 2. The
// quadrantInLcp(node, 0) returns 01 (1010011)
//                                        ^^
// quadrantInLcp(node, 1) returns 10 (1010011)
//                                      ^^
// quadrantInLcp(node, 2) returns 10 (1010011)
//                                    ^^
int quadrantInLcp(BrtNode* brt_node, const int i) {
  const int mask = (DIM == 2) ? 3 : 7;
  /* if (DIM > 3) */
  /*   throw logic_error("BrtNode::oct_nodes not yet supported for D>3"); */
  const int rem = brt_node->lcp_length % DIM;
  int rshift = i * DIM + rem;
  //return (brt_node->lcp >> rshift) & mask;
	BigUnsigned temp;
	BigUnsigned tempb;
	BigUnsigned buMask;
	initBlkBU(&buMask, mask);
	shiftBURight(&temp, &brt_node->lcp, rshift);
	andBU(&tempb, &temp, &buMask);
	return getBUBlock(&tempb, 0); //Not sure if this is right. I'm assuming the quadrant is at DIM bits
}

/*
  Binary radix to Octree
*/
//gid is between 0 and the octree size excluding 0
void brt2octree( const int gid, __global BrtNode* I, __global volatile OctNode* octree, __global unsigned int* local_splits, __global unsigned int* prefix_sums, const int n, const int octree_size) {
  //If I have a local split
  if (local_splits[gid] > 0) {
    const int mySplits = local_splits[gid];
    BrtNode myBRTNode = I[gid];
    
    // Current octree node index initializes as right before me
    int currentOctIndx = prefix_sums[gid-1];
    int quadrant;

    //For each of my splits, initialize an octnode
    for (int i = 0; i < mySplits -1; ++i) {
      currentOctIndx++;
      quadrant = quadrantInLcp(&myBRTNode, i);
      octree[currentOctIndx].children[quadrant] = currentOctIndx - 1;
      octree[currentOctIndx - 1].parent = currentOctIndx;
      octree[currentOctIndx - 1].level = (myBRTNode.lcp_length / DIM) - i;
      if (currentOctIndx > -1) 
        octree[currentOctIndx].leaf &= ~leaf_masks[quadrant];
      else 
        octree[currentOctIndx].leaf |= leaf_masks[quadrant];
    }

    //Find my first ancestor containing a split...
    int parentBRTIndx = I[gid].parent;
    while (local_splits[parentBRTIndx] == 0) 
      parentBRTIndx = I[parentBRTIndx].parent;
    
    //The octparent is either root or the node index before the parent BRT node
    int parentOctIndx = (parentBRTIndx == 0) ? 0 : prefix_sums[parentBRTIndx - 1];

    //the parent octnode's child at my quadrant is me.
    quadrant = quadrantInLcp(&myBRTNode, mySplits - 1);
    octree[parentOctIndx].children[quadrant] = currentOctIndx;
    octree[currentOctIndx].parent = parentOctIndx;
    octree[currentOctIndx].level = (myBRTNode.lcp_length / DIM) - mySplits + 1;

    if (currentOctIndx > -1) {
      #ifndef  __OPENCL_VERSION__
        octree[parentOctIndx].leaf &= ~leaf_masks[quadrant];
      #else
        atomic_and(&octree[parentOctIndx].leaf, ~leaf_masks[quadrant]);
      #endif
    }
    else {
      #ifndef  __OPENCL_VERSION__
        octree[parentOctIndx].leaf |= leaf_masks[quadrant];
      #else
        atomic_or(&octree[parentOctIndx].leaf, leaf_masks[quadrant]);
      #endif
    }
  }
}
void brt2octree_init(const int brt_i, __global OctNode* octree ) {
  octree[brt_i].leaf = 15;
  for (int i = 0; i < (1 << DIM); ++i) {
    octree[brt_i].children[i] = -1;
  }
}
void brt2octree_kernel(__global BrtNode* I, __global OctNode* octree, __global unsigned int* local_splits, __global unsigned int* prefix_sums, const int n) {
  const int octree_size = prefix_sums[n-2];
  // Initialize octree - needs to be done in parallel
  for (int i = 0; i < octree_size; ++i)
    brt2octree_init( i, octree);
  for (int brt_i = 1; brt_i < n-1; ++brt_i)
    brt2octree( brt_i, I, octree, local_splits, prefix_sums, n, octree_size);
}

// gid is between 0 and 4/8X the octree size.
void ComputeLeaves(__global OctNode *octree, __global Leaf *leaves, __global cl_int *leafPredicates, cl_int octreeSize, cl_int gid) {
  int parentIndex = gid / 4;
  int leafIndex = gid % 4;
  OctNode n = octree[parentIndex];
  if (parentIndex < octreeSize && n.leaf & (1 << leafIndex)) {
    leaves[gid].parent = parentIndex;
    leaves[gid].zIndex = leafIndex;
    leafPredicates[gid] = 1;
  }
  else {
    leaves[gid].parent = -1;
    leaves[gid].zIndex = -1;
    leafPredicates[gid] = 0;
  }

}

void LeafDoubleCompact(__global Leaf *inputBuffer, __global Leaf *resultBuffer, __global unsigned int *lPredicateBuffer, __global unsigned int *leftBuffer, const cl_int size, const int gid)
{
  int a = leftBuffer[gid];
  int b = leftBuffer[size - 2];
  int c = lPredicateBuffer[gid];
  int e = lPredicateBuffer[size - 1];

  //Check out http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html figure 39-14
  int t = gid - a + (e + b);
  int d = (!c) ? t : a - 1;

  //This really suffers from poor coalescing
#ifdef __OPENCL_VERSION__
  barrier(CLK_GLOBAL_MEM_FENCE);
#endif
  resultBuffer[d] = inputBuffer[gid];
}

#ifndef __OPENCL_VERSION__
  #undef __local
  #undef __global
#endif
