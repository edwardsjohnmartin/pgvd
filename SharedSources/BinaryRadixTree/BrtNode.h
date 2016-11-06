#pragma once

#ifdef __OPENCL_VERSION__
#include "./SharedSources/BigUnsigned/BigUnsigned.h"
#else
#include "../BigUnsigned/BigUnsigned.h"
#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#else
#include "CL/cl_platform.h"
#endif
#endif

typedef struct BrtNode {
  // left child (right child = left+1)
  cl_int left;
  // Whether the left (resp. right) child is a leaf or not
  bool left_leaf, right_leaf;
  // The longest common prefix
  BigUnsigned lcp;
  // Number of bits in the longest common prefix
  cl_int lcp_length;
  // Secondary - computed in a second pass
  cl_int parent;
} BrtNode;

inline bool compareBrtNode(BrtNode* x, BrtNode* y) {
  if (weakEqualsBU(x->lcp, y->lcp) != true) return false;
  if (x->lcp_length != y->lcp_length) return false;
  if (x->left != y->left) return false;
  if (x->left_leaf != y->left_leaf) return false;
  if (x->parent != y->parent) return false;
  if (x->right_leaf != y->right_leaf) return false;
  return true;
}

