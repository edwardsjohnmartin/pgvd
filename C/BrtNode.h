#ifndef __BRT_NODE_H__
#define __BRT_NODE_H__

#ifdef __OPENCL_VERSION__
#include ".\opencl\C\BigUnsigned.h"
#else
#include "BigUnsigned.h"
#endif

typedef struct BrtNode {
  // left child (right child = left+1)
  int left;
  // Whether the left (resp. right) child is a leaf or not
  bool left_leaf, right_leaf;
  // The longest common prefix
  BigUnsigned lcp;
  // Number of bits in the longest common prefix
  int lcp_length;
  // Secondary - computed in a second pass
  int parent;
} BrtNode;

#endif
