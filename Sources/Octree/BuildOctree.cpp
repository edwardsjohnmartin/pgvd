#include "Octree/BuildOctree.h"
#include "ParallelAlgorithms/ParallelAlgorithms.h"

#ifndef  OpenCL 
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#define __local
#define __global
#define barrier(a)
#endif

void ComputeLocalSplits(__global cl_int* splits, __global BrtNode* node, const int gid) {
  const int currentLenPerDim = node[gid].lcp.len / DIM;
  const int left = node[gid].left;
  const int right = left + 1;
  if (!node[gid].left_leaf) {
    splits[left] = node[left].lcp.len/ DIM - currentLenPerDim;
  }
  if (!node[gid].right_leaf) {
    splits[right] = node[right].lcp.len/ DIM - currentLenPerDim;
  }
}

// stores total times octree should be split from a parent to a child in BRT.
// eg 2D child with lcplen 8 & parent lcpLen 4 =? child reps 2 octree splits.
void ComputeLocalSplits_SerialKernel(__global cl_int* local_splits, __global BrtNode* node, const cl_int size) {
  if (size > 0) {
    local_splits[0] = 1 + node[0].lcp.len/ DIM;
  }
  for (int i = 0; i < size - 1; ++i) {
    ComputeLocalSplits(local_splits, node, i);
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
int getQuadrantFromBrt(BrtNode* brt_node, const int i) {
  const int mask = (DIM == 2) ? 3 : 7;
  const int rem = brt_node->lcp.len% DIM;
  int rshift = i * DIM + rem;
  BigUnsigned temp;
  BigUnsigned tempb;
  BigUnsigned buMask;
  initBlkBU(&buMask, mask);
  shiftBURight(&temp, &brt_node->lcp.bu, rshift);
  andBU(&tempb, &temp, &buMask);
  return getBUBlock(&tempb, 0);
}

int getQuadrantFromLCP(BigUnsigned lcp, cl_int lcpLen, cl_int i) {
  const int mask = (DIM == 2) ? 3 : 7;
  const int rem = lcpLen % DIM;
  int rshift = i * DIM + rem;
  BigUnsigned temp;
  BigUnsigned tempb;
  BigUnsigned buMask;
  initBlkBU(&buMask, mask);
  shiftBURight(&temp, &lcp, rshift);
  andBU(&tempb, &temp, &buMask);
  return getBUBlock(&tempb, 0);
}

/*
  Binary radix to Octree
*/
//gid is between 0 and the octree size excluding 0
void brt2octree(__global BrtNode* BRT, const cl_int totalBrtNodes, __global OctNode* octree, 
  const cl_int totalOctNodes, __global cl_int* local_splits, __global cl_int* prefix_sums, 
  __global cl_int* flags, const cl_int gid) {
  BrtNode brtNode = BRT[gid];
  OctNode n;

  /* Get total number of nodes to generate, and where to put them */
  const cl_int mySplits = local_splits[gid];
  const cl_int startIndx = prefix_sums[gid - 1];

  /* Build first node in the list */
  for (int i = 0; i < mySplits; ++i) {
    /* If this is the first node, we need to find the first BRT ancestor with a split, and use 
       that BRT Node's last split as the parent.
       If this isn't the first node, the parent is just the previously generated node. */

    if (i == 0) {
      /* Find the first ancestor containing a split... */
      int parentBRTIndx = brtNode.parent;
      while (local_splits[parentBRTIndx] == 0)
        parentBRTIndx = BRT[parentBRTIndx].parent;
      n.parent = (parentBRTIndx == 0) ? 0 : prefix_sums[parentBRTIndx - 1] + local_splits[parentBRTIndx] - 1;
    }
    else { n.parent = startIndx + i - 1; }

    n.level = ((brtNode.lcp.len/ DIM) - mySplits) + i + 1;
    n.quadrant = getQuadrantFromBrt(&brtNode, mySplits - 1 - i);

    /* Write all fields except child to avoid race conditions */
    octree[startIndx + i].level = n.level;
    octree[startIndx + i].parent = n.parent;
    octree[startIndx + i].quadrant = n.quadrant;

    /* Add this first node to its parent */
    octree[n.parent].children[n.quadrant] = startIndx + i;
  }
}
void brt2octree_init(__global OctNode* octree, const int gid) {
  /* Start all nodes as having only leaves for children. */
  OctNode n;
  for (int i = 0; i < (1 << DIM); ++i) n.children[i] = -1;

  /* Initialize the root octnode */
  if (gid == 0) {
    n.parent = n.level = n.quadrant = -1;
    n.level = 0;
  }
  octree[gid] = n;
}
//void brt2octree_kernel(__global BrtNode* I, __global OctNode* octree, __global cl_int* local_splits, __global cl_int* prefix_sums, const cl_int n) {
//  const int octree_size = prefix_sums[n - 2];
//  // Initialize octree - needs to be done in parallel
//  for (int i = 0; i < octree_size; ++i)
//    brt2octree_init(octree, i);
//  for (int brt_i = 1; brt_i < n - 1; ++brt_i)
//    brt2octree(I, ,brt_i, I, octree, local_splits, prefix_sums, n, octree_size);
//}

// gid is between 0 and 4/8X the octree size.
void ComputeLeaves(__global OctNode *octree, __global Leaf *leaves, __global cl_int *leafPredicates, cl_int octreeSize, cl_int gid) {
  int parentIndex = gid / 4;
  int leafIndex = gid % 4;
  OctNode n = octree[parentIndex];
	int isLeaf = n.children[leafIndex] == -1;
  Leaf L;
  if (isLeaf) {
    L.parent = parentIndex;
    L.quadrant = leafIndex;
  }
  else {
    L.parent = -1;
    L.quadrant = -1;
  }
  leaves[gid] = L;
  leafPredicates[gid] = isLeaf;
}

void LeafDoubleCompact(__global Leaf *inputBuffer, __global Leaf *resultBuffer, __global cl_int *lPredicateBuffer, __global cl_int *leftBuffer, const cl_int size, const int gid)
{
  int a = leftBuffer[gid];
  int b = leftBuffer[size - 2];
  int c = lPredicateBuffer[gid];
  int e = lPredicateBuffer[size - 1];

  //Check out http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html figure 39-14
  int t = gid - a + (e + b);
  int d = (!c) ? t : a - 1;

  //This really suffers from poor coalescing
#ifdef OpenCL
  barrier(CLK_GLOBAL_MEM_FENCE);
#endif
  resultBuffer[d] = inputBuffer[gid];
}

inline void getOctNodeLCP(BigUnsigned* lcp, cl_int *lcpLen, __global OctNode *octree, int gid) {
  OctNode octnode = octree[gid];
  initBlkBU(lcp, 0);
  cl_int level = 0;
  while (octnode.parent != -1) {
    BigUnsigned temp;
    initBlkBU(&temp, octnode.quadrant);
    shiftBULeft(&temp, &temp, DIM * level);
    orBU(lcp, lcp, &temp);
    level++;
    octnode = octree[octnode.parent];
  }
  *lcpLen = level * DIM;
}

//Note, assumes root exists AKA octreeSize > 0
//LCP does not include leaf.
inline int searchForOctNode(BigUnsigned lcp, int lcpLength, __global OctNode *octree) {
  if (lcpLength == 0) return 0;
  OctNode current = octree[0];
  int index = -1;
  for (int i = 0; i < lcpLength / DIM; ++i) {
    int quadrant = getQuadrantFromLCP(lcp, lcpLength, i);
    if ((current.leaf & quadrant) != 0) return -1; 
    index = current.children[quadrant];
    current = octree[current.children[quadrant]];
  }
  return index;
}

// Use Z-order to determine if the current node in the new octree already exists in
// the old one. If it's a dup, then put the current node's original address in the
// predicate buffer. Otherwise, predicate with -1, meaning delete this node.
void PredicateDuplicateNodes(__global OctNode *origOT, __global OctNode *newOT, __global cl_int *predicates, int newOTSize, int gid) {
  cl_int index;
  {
    BigUnsigned LCP;
    cl_int lcpLen = 0;
    getOctNodeLCP(&LCP, &lcpLen, newOT, gid);
    index = searchForOctNode(LCP, lcpLen, origOT);
  }

  if (index == -1) {
    predicates[gid] = -1; //Do not compact me.
  }
  else {
    predicates[gid] = index;
  }

  //OctNode current = origOT[gid];
  //current.parent++;
  //origOT[gid] = current;
  //for (int i = 0; i < DIM; ++i) {
  //  //When I perish, have the original node adopt my children. ;(
  //  //newOT[gid].parent = current.level;
  //  newOT[gid] = current;
  //  barrier(CLK_GLOBAL_MEM_FENCE);
  //}


  //else {
  //  barrier(CLK_GLOBAL_MEM_FENCE)
  //  // current.children[0];
  //  //for (int i = 0; i < DIM; ++i) {
  //  //  if ((current.leaf & (1 << i)) == 0) {
  //  //    newOT[gid].parent = 1;
  //  ////    newOT[current.children[i]].parent = index;
  //  //  }
  //  //}
  //}
}

#ifndef OpenCL
#undef __local
#undef __global
#endif
