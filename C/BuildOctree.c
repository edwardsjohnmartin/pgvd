#ifdef  __OPENCL_VERSION__ 
  #include ".\opencl\C\BuildOctree.h"
  #include ".\opencl\C\ParallelAlgorithms.h"
#else
  #include <stdbool.h>
  #include <stdio.h>
  #include <math.h>
  #include "BuildOctree.h"
  #include "ParallelAlgorithms.h"
#endif

#ifndef __OPENCL_VERSION__
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
int quadrantInLcp(const BrtNode* brt_node, const int i) {
  const int mask = (DIM == 2) ? 3 : 7;
  /* if (DIM > 3) */
  /*   throw logic_error("BrtNode::oct_nodes not yet supported for D>3"); */
  const int rem = brt_node->lcp_length % DIM;
  const int rshift = i * DIM + rem;
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

void brt2octree_end(const int brt_i, __global BrtNode* I, __global OctNode* octree, __global unsigned int* local_splits, __global unsigned int* prefix_sums, const int n, const int octree_size) {

}
void brt2octree( const int brt_i, __global BrtNode* I, __global volatile OctNode* octree, __global unsigned int* local_splits, __global unsigned int* prefix_sums, const int n, const int octree_size) {
  if (local_splits[brt_i] > 0) {
    // m = number of local splits
    const int numSplits = local_splits[brt_i];
    BrtNode brt_node;
    brt_node = I[brt_i];
    
    // Current octree node index
    int currentNode;
    if (brt_i == 0) {
      currentNode = 0;
    }
    else {
      currentNode = prefix_sums[brt_i-1]; //current node might be race condition. prefix sums are not unique.
    }
    for (int i = 0; i < numSplits -1; ++i) {
      const int oct_parent = currentNode+1; //oct_parent is not guaranteed to be unique!
      const int onode = quadrantInLcp(&brt_node, i);
      //set_child(&octree[oct_parent], onode, currentNode);
//No race conditions up to this point
      octree[oct_parent].children[onode] = currentNode;  //Children modified! can't be read from w.o. race condition.
      if (currentNode > -1) {
        octree[oct_parent].leaf &= ~leaf_masks[onode]; //leaf modified! can't be read from w.o. race condition.
      }
      else {
        octree[oct_parent].leaf |= leaf_masks[onode];
      }
      currentNode = oct_parent;
    }
    int brt_parent = I[brt_i].parent;
    while (local_splits[brt_parent] == 0) {
      brt_parent = I[brt_parent].parent;
    }
    int oct_parent;
    if (brt_parent == 0) {
      oct_parent = 0;
    }
    else {
      oct_parent = prefix_sums[brt_parent-1];
    }

    //set_child(&octree[oct_parent], quadrantInLcp(brt_node, m-1), currentNode);
    int temp = quadrantInLcp(&brt_node, numSplits - 1);
    octree[oct_parent].children[temp] = currentNode;
    if (currentNode > -1) {
      #ifndef  __OPENCL_VERSION__
        octree[oct_parent].leaf &= ~leaf_masks[temp];
      #else
        atomic_and(&octree[oct_parent].leaf, ~leaf_masks[temp]);
      #endif
    }
    else {
      #ifndef  __OPENCL_VERSION__
        octree[oct_parent].leaf |= leaf_masks[temp];
      #else
        atomic_or(&octree[oct_parent].leaf, leaf_masks[temp]);
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

#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#endif

