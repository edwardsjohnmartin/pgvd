#ifndef __OCT_NODE_H__
#define __OCT_NODE_H__
// An octree node is an internal node of the octree. An octree cell
// is a general term that refers to both internal nodes and leaves.

#ifndef __OPENCL_VERSION__
#define __local
#define __global
#endif

#ifndef  __OPENCL_VERSION__ 
static const int leaf_masks[] = { 1, 2, 4, 8 };
#include "dim.h"
#include "BigUnsigned.h"
#else
__constant int leaf_masks[] = { 1, 2, 4, 8 };
#include "./opencl/C/dim.h"
#include "./opencl/C/BigUnsigned.h"
#endif


// You must call init_OctNode()!
typedef struct OctNode {

#ifdef __cplusplus
  const int& operator[](const int i) const {
    return children[i];
  }
#endif // __cplusplus

  int children[1<<DIM];
  int leaf;
  int level;
  int parent;
} OctNode;

static inline void init_OctNode(struct OctNode* node) {
  node->leaf = 15;
  for (int i = 0; i < (1<<DIM); ++i) {
    node->children[i] = -1;
  }
}

static inline void set_child(struct OctNode* node, const int octant, const int child) {
  node->children[octant] = child;
  if (child > -1) {
    node->leaf &= ~leaf_masks[octant];
  } else {
    node->leaf |= leaf_masks[octant];
  }
}

static inline bool is_leaf(const struct OctNode* node, const int i) {
  return node->leaf & leaf_masks[i];
}

static inline void set_data(struct OctNode* node, const int octant, const int data) {
  node->children[octant] = data;
}
inline bool compareOctNode(OctNode* first, OctNode* second) {
  for (int i = 0; i < 1 << DIM; ++i)
    if (first->children[i] != second->children[i]) return false;
  if (first->leaf != second->leaf) return false;
  return true;
}

inline int getOctNode(BigUnsigned lcp, int lcpLength, __global OctNode *octree) {
  BigUnsigned mask;
  BigUnsigned result;
  int numLevels = lcpLength / DIM;
  int isOdd = ((lcpLength & 1) == 1) ? 1 : 0;
  int currentIndex = 0;
  int parentIndex = 0;
  int childIndex = 0;
  OctNode node = octree[currentIndex];
  for (int i = 0; i < numLevels; ++i) {
    //Get child index
    if (lcp.len != 0) {
      BigUnsigned mask;
      BigUnsigned result;
      int shiftAmount = (numLevels - i - 1) * DIM + isOdd;
      initBlkBU(&mask, ((DIM == 2) ? 3 : 7));
      shiftBULeft(&mask, &mask, shiftAmount);
      andBU(&result, &mask, &lcp);
      shiftBURight(&result, &result, shiftAmount);
      childIndex = result.blk[0];
    }

    currentIndex = node.children[childIndex];
    if (currentIndex == -1) {
      //The current LCP sits within a leaf node.
      return parentIndex;
    }
    node = octree[currentIndex];
    parentIndex = currentIndex;
  }
  //The LCP refers to an internal node.
  return currentIndex;
}

//#ifdef __cplusplus
//
//inline std::ostream& operator<<(std::ostream& out, const OctNode& node) {
//  for (int i = 0; i < 1<<DIM; ++i) {
//    out << node.children[i] << " ";
//  }
//  out << static_cast<int>(node.leaf);
//  return out;
//}
//
//inline std::istream& operator>>(std::istream& in, OctNode& node) {
//  for (int i = 0; i < 1<<DIM; ++i) {
//    in >> node.children[i];
//  }
//  int leaf;
//  in >> leaf;
//  node.leaf = static_cast<unsigned char>(leaf);
//  return in;
//}
//
//inline std::ostream& operator<<(
//    std::ostream& out, const std::vector<OctNode>& octree) {
//  out << octree.size() << " ";
//  for (const OctNode& node : octree) {
//    out << node << " ";
//  }
//  return out;
//}
//
//inline std::istream& operator>>(
//    std::istream& in, std::vector<OctNode>& octree) {
//  int n;
//  in >> n;
//  octree.resize(n);
//  for (int i = 0; i < n; ++i) {
//    in >> octree[i];
//  }
//  return in;
//}
//
//#endif // __cplusplus

#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#endif

#endif
