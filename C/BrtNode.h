#ifndef __BRT_NODE_H__
#define __BRT_NODE_H__

#ifndef __OPENCL_VERSION__
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

  // left child (right child = left+1)
  int left1;
  // Whether the left (resp. right) child is a leaf or not
  bool left_leaf1, right_leaf1;
  // The longest common prefix
  BigUnsigned lcp1;
  // Number of bits in the longest common prefix
  int lcp_length1;
  // Secondary - computed in a second pass
  int parent1;
} BrtNode;

#endif
