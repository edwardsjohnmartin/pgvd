#include "./bool.h"
#include "BuildOctree.h"

#include <stdio.h>
#include <math.h>

//------------------------------------------------------------
// Utility functions
//------------------------------------------------------------

int sign(const int i) {
  return (i<0) ? -1 : ((i>0) ? +1 : 0);
}

// Longest common prefix
//
// Suppose mbits = 6, then morton code is
//   ______
// 00011010
//
// Suppose length = 3, then lcp (masked) is
//   ___
// 00011000
//
// Now shift, and lcp is
//      ___
// 00000011
void compute_lcp(BigUnsigned *lcp, BigUnsigned *value, const int length, const struct Resln* resln) {
	BigUnsigned mask;
	initBlkBU(&mask, 0);
	BigUnsigned one;
	initBlkBU(&one, 1);
	BigUnsigned temp;
	initBU(&temp);
  for (int i = 0; i < length; ++i) {
    //mask |= (one << (resln->mbits - 1 - i));
		shiftBULeft(&temp, &one, (resln->mbits - 1 - i));
		orBU(&mask, &mask, &temp);
  }
  //const BigUnsigned lcp = (value & mask) >> (resln->mbits - length);
	andBU(&temp, value, &mask);
	shiftBURight(lcp, &temp, resln->mbits - length);
}

// Longest common prefix, denoted \delta in karras2014
int compute_lcp_length_impl(BigUnsigned* a, BigUnsigned* b, const struct Resln* resln) {
	BigUnsigned one;
	initBlkBU(&one, 1);
	BigUnsigned tempa, tempb;
  BigUnsigned mask;
  for (int i = resln->mbits-1; i >= 0; --i) {
		//BigUnsigned mask = one << i;
		shiftBULeft(&mask, &one, i);
		//if ((a & mask) != (b & mask)) {
		andBU(&tempa, a, &mask);
		andBU(&tempb, b, &mask);
		if(compareBU(&tempa, &tempb) != 0){
      return resln->mbits - i - 1;
    }
  }
  return resln->mbits;
}

int compute_lcp_length(const int i, const int j,
                       BigUnsigned* _mpoints, const struct Resln* _resln) {
  return compute_lcp_length_impl(&_mpoints[i], &_mpoints[j], _resln);
}

// Given a lcp, returns the i'th quadrant starting from the most local.
// Suppose node.lcp is 1010011 and DIM == 2. The
// quadrantInLcp(node, 0) returns 01 (1010011)
//                                        ^^
// quadrantInLcp(node, 1) returns 10 (1010011)
//                                      ^^
// quadrantInLcp(node, 2) returns 10 (1010011)
//                                    ^^
int quadrantInLcp(const struct BrtNode* brt_node, const int i) {
  static const int mask = (DIM == 2) ? 3 : 7;
  assert(DIM <= 3);
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

//------------------------------------------------------------
//------------------------------------------------------------
// Kernels
//------------------------------------------------------------
//------------------------------------------------------------

void build_brt_kernel(
    const int i, struct BrtNode* I, struct BrtNode* L,
    const BigUnsigned* mpoints, const int n,
    const struct Resln* resln) {
  // Determine direction of the range (+1 or -1)
  int d;
  if (i == 0) {
    d = 1;
  } else {
    const int l_pos = compute_lcp_length(i, i+1, mpoints, resln);
    const int l_neg = compute_lcp_length(i, i-1, mpoints, resln);
    d = sign(l_pos - l_neg);
  }
  // Compute upper bound for the length of the range
  int l;
  if (i == 0) {
    l = n-1;
  } else {
    const int lcp_min = compute_lcp_length(i, i-d, mpoints, resln);
    int l_max = 2;
    while (i+l_max*d >= 0 && i+l_max*d <= n-1 && compute_lcp_length(i, i+l_max*d, mpoints, resln) > lcp_min) {
      l_max = l_max << 1;
    }
    // Find the other end using binary search.
    // In some cases, the search can go right off the end of the array.
    // l_max likely is beyond the end of the array, but we need it to be
    // since it's a power of 2. So define a max length that we call l_cutoff.
    const int l_cutoff = (d==-1) ? i : n - i - 1;
    l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
      if (l + t <= l_cutoff) {
        if (compute_lcp_length(i, i+(l+t)*d, mpoints, resln) > lcp_min) {
          l = l + t;
        }
      }
    }
  }

  // j is the index of the other end of the range. In other words,
  // range = [i, j] or range = [j, i].
  const int j = i + l * d;
  // Find the split position using binary search
  const int lcp_node = compute_lcp_length(i, j, mpoints, resln);

  const int s_cutoff = (d==-1) ? i - 1 : n - i - 2;
  int s = 0;
  for (int den = 2; den < 2*l; den *= 2) {
    const int t = (int)(ceil(l/(double)den));
    if (s + t <= s_cutoff) {
      if (compute_lcp_length(i, i+(s+t)*d, mpoints, resln) > lcp_node) {
        s = s + t;
      }
    }
  }
  const int split = i + s * d + fmin(d, 0);
  // Output child pointers
  I[i].left = split;
  I[i].left_leaf = (fmin(i, j) == split);
  I[i].right_leaf = (fmax(i, j) == split+1);
  compute_lcp(&I[i].lcp,&mpoints[i], lcp_node, resln);
  I[i].lcp_length = lcp_node;
}

void compute_local_splits_kernel(
    const int i, int* local_splits, struct BrtNode* I, const int n) {
  const int local = I[i].lcp_length / DIM;
  const int left = I[i].left;
  const int right = left+1;
  if (!I[i].left_leaf) {
    local_splits[left] = I[left].lcp_length/DIM - local;
  }
  if (!I[i].right_leaf) {
    local_splits[right] = I[right].lcp_length/DIM - local;
  }
}

void brt2octree_kernel(
    const int brt_i,
    const struct BrtNode* I, struct OctNode* octree, const int* local_splits,
    const int* prefix_sums, const int n, const int octree_size) {
  if (local_splits[brt_i] > 0) {
    // m = number of local splits
    const int m = local_splits[brt_i];
    const struct BrtNode* brt_node = &I[brt_i];
    // Current octree node index
    int oct_i = prefix_sums[brt_i];
    for (int j = 0; j < m-1; ++j) {
      const int oct_parent = oct_i+1;
      const int onode = quadrantInLcp(brt_node, j);
      set_child(&octree[oct_parent], onode, oct_i);
      oct_i = oct_parent;
    }
    int brt_parent = I[brt_i].parent;
    while (local_splits[brt_parent] == 0) {
      brt_parent = I[brt_parent].parent;
    }
    const int oct_parent = prefix_sums[brt_parent];
    assert(brt_parent >= 0 && brt_parent < n);
    assert(oct_parent >= 0 && oct_parent < octree_size);
    set_child(&octree[oct_parent], quadrantInLcp(brt_node, m-1), oct_i);
  }
}

//------------------------------------------------------------
//------------------------------------------------------------
// Calling functions
//------------------------------------------------------------
//------------------------------------------------------------

// Needs to be implemented
void sort_points(BigUnsigned* mpoints, const int n) {
  /* sort(mpoints, mpoints + n); */
  printf("sort_points() not implemented\n");
}

// JME: Haven't split this out into a kernel...
int unique_points(BigUnsigned* mpoints, BigUnsigned* dest, const int n) {
  if (n == 0) return 0;

  dest[0] = mpoints[0];
  int idx = 1;
  for (int i = 1; i < n; ++i) {
    //if (mpoints[i] != mpoints[i-1]) {
		if (compareBU(&mpoints[i], &mpoints[i - 1]) != 0) {
      dest[idx++] = mpoints[i];
    }
  }
  return idx;
}

void build_brt(
    struct BrtNode* I, struct BrtNode* L,
    const BigUnsigned* mpoints, const int n,
    const struct Resln* resln) {
  // Note that it loops only n-1 times.
  for (int i = 0; i < n-1; ++i) {
    build_brt_kernel(i, I, L, mpoints, n, resln);
  }
}

// JME: Haven't split this into a kernel
void set_brt_parents(struct BrtNode* I, const int n) {
  if (n > 0)
    I[0].parent = -1;
  for (int i = 0; i < n-1; ++i) {
    const int left = I[i].left;
    const int right = left+1;
    if (!I[i].left_leaf) {
      I[left].parent = i;
    }
    if (!I[i].right_leaf) {
      I[right].parent = i;
    }
  }
}

// local_splits stores how many times the octree needs to be
// split from a parent to a child in the brt. For example, in 2D if a child
// has an lcp_length of 8 and the parent has lcp_length of 4, then
// the child represents two octree splits.
void compute_local_splits(
    int* local_splits, struct BrtNode* I, const int n) {
  if (n > 0) {
    local_splits[0] = 1 + I[0].lcp_length / DIM;
  }
  for (int i = 0; i < n-1; ++i) {
    compute_local_splits_kernel(i, local_splits, I, n);
  }
}

// JME: haven't split this into a kernel
void compute_prefix_sums(const int* src, int* prefix_sums, const int n) {
  prefix_sums[0] = 0;
  for (int i = 1; i < n; ++i) {
    prefix_sums[i] = src[i-1];
  }
  for (int i = 1; i < n; ++i) {
    prefix_sums[i] = prefix_sums[i-1] + prefix_sums[i];
  }
}

void brt2octree(
    const struct BrtNode* I, struct OctNode* octree, const int* local_splits,
    const int* prefix_sums, const int n) {
  const int octree_size = prefix_sums[n-1];
  // Initialize octree - needs to be done in parallel
  for (int i = 0; i < octree_size; ++i) {
    init_OctNode(&octree[i]);
  }
  for (int brt_i = 1; brt_i < n-1; ++brt_i) {
    brt2octree_kernel(
        brt_i, I, octree, local_splits, prefix_sums, n, octree_size);
  }
}


