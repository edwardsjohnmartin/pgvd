#ifndef __BUILD_OCTREE_H__
#define __BUILD_OCTREE_H__

#include "../Resln.h"
#include "../OctNode.h"

int sign(const int i);

void compute_lcp(BigUnsigned *lcp, BigUnsigned *value, const int length, const struct Resln* resln);

// int compute_lcp_length_impl(const BigUnsigned a, const BigUnsigned b, const struct Resln* resln);

int compute_lcp_length(const int i, const int j,
                       BigUnsigned* _mpoints, const struct Resln* _resln);

struct BrtNode {
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
};

int quadrantInLcp(const struct BrtNode* brt_node, const int i);

void sort_points(BigUnsigned* mpoints, const int n);

int unique_points(BigUnsigned* mpoints, BigUnsigned* dest, const int n);

void build_brt(
    struct BrtNode* I, struct BrtNode* L,
    const BigUnsigned* mpoints, const int n,
    const struct Resln* resln);

void set_brt_parents(struct BrtNode* I, const int n);

void compute_local_splits(
    int* local_splits, struct BrtNode* I, const int n);

void compute_prefix_sums(const int* src, int* prefix_sums, const int n);

void brt2octree(
    const struct BrtNode* I, struct OctNode* octree, const int* local_splits,
    const int* prefix_sums, const int n);

#endif
