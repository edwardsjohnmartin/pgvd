#ifndef __OPENCL_VERSION__ 
extern "C" {
  #include "BuildBRT.h"
}
#endif

// UTILITY FUNCTIONS
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#ifndef __OPENCL_VERSION__
int sign(const int i) {
  return (i<0) ? -1 : ((i>0) ? +1 : 0);
}
#endif

// LEAST COMMON PREFIX CALCULATIONS (\delta in karras2014)
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
void compute_lcp(__global BigUnsigned *lcp, __global BigUnsigned *value, const int length, int mbits) {
	BigUnsigned mask;
	initBlkBU(&mask, 0);
	BigUnsigned one;
	initBlkBU(&one, 1);
  BigUnsigned temp;
  BigUnsigned privateValue, privateLcp;
  privateValue = *value;
  privateLcp = *lcp;
	initBU(&temp);
  for (int i = 0; i < length; ++i) {
		shiftBULeft(&temp, &one, (mbits - 1 - i));
		orBU(&mask, &mask, &temp);
  }
	andBU(&temp, &privateValue, &mask);
	shiftBURight(&privateLcp, &temp, mbits - length);
  *value = privateValue;
  *lcp = privateLcp;
}

int compute_lcp_length(int i, int j, __global BigUnsigned* _mpoints, int mbits) {
  BigUnsigned one;
  initBlkBU(&one, 1);
  BigUnsigned tempa, tempb, tempc, tempd;
  BigUnsigned mask;
  tempc = _mpoints[i];
  tempd = _mpoints[j];

  for (int k = mbits - 1; k >= 0; --k) {
    //BigUnsigned mask = one << i;
    shiftBULeft(&mask, &one, k);
    //if ((a & mask) != (b & mask)) {
    tempc = _mpoints[i];
    tempd = _mpoints[j];
    andBU(&tempa, &tempc, &mask);
    andBU(&tempb, &tempd, &mask);

    if (compareBU(&tempa, &tempb) != 0) {
      return mbits - k - 1;
    }
  }
  return mbits;
}

void BuildBinaryRadixTree( __global BrtNode *I, __global BrtNode* L, __global BigUnsigned* mpoints, int mbits, int size, const unsigned int gid)
{
  //n-1 internal nodes.
  if (gid < size-1) {
    // Determine direction of the range (+1 or -1)
    int d;
    if (gid == 0) 
      d = 1;
    else {
      int l_pos = compute_lcp_length(gid, gid + 1, mpoints, mbits);
      int l_neg = compute_lcp_length(gid, gid - 1, mpoints, mbits);
      d = ((l_pos - l_neg > 0) - (l_pos - l_neg < 0)); //sign
    }

    // Compute upper bound for the length of the range
    int l;
    if (gid == 0) {
      l = size-1;
    } else {
      const int lcp_min = compute_lcp_length(gid, gid-d, mpoints, mbits);
      int l_max = 2;
      while ( gid + l_max * d >= 0 && 
              gid + l_max * d <= size - 1 && 
              compute_lcp_length( gid, gid + l_max * d, mpoints, mbits) > lcp_min) 
      {
        l_max = l_max << 1;
      }
      // Find the other end using binary search.
      // In some cases, the search can go right off the end of the array.
      // l_max likely is beyond the end of the array, but we need it to be
      // since it's a power of 2. So define a max length that we call l_cutoff.
      const int l_cutoff = (d==-1) ? gid : size - gid - 1;
      l = 0;
      for (int t = l_max >> 1; t >= 1; t >>= 1) {
        if (l + t <= l_cutoff) {
          if (compute_lcp_length(gid, gid+(l+t)*d, mpoints, mbits) > lcp_min) {
            l = l + t;
          }
        }
      }
    }

    // j is the index of the other end of the range. In other words,
    // range = [i, j] or range = [j, i].
    const int j = gid + l * d;
    // Find the split position using binary search
    const int lcp_node = compute_lcp_length(gid, j, mpoints, mbits);
    const int s_cutoff = (d==-1) ? gid - 1 : size - gid - 2;
    int s = 0;
    for (int den = 2; den < 2*l; den *= 2) {
      const int t = (int)((l + (float)den - 1) / den);// ceil(l / (float)den));
      if (s + t <= s_cutoff) {
        if (compute_lcp_length(gid, gid+(s+t)*d, mpoints, mbits) > lcp_node) {
          s = s + t;
        }
      }
    }
    const int split = gid + s * d + MIN(d, 0);
    // Output child pointers
    I[gid].left = split;
    I[gid].left_leaf = (MIN(gid, j) == split);
    I[gid].right_leaf = (MAX(gid, j) == split+1);
    compute_lcp(&I[gid].lcp, &mpoints[gid], lcp_node, mbits);
    I[gid].lcp_length = lcp_node;

    //Set parents
    if (gid == 0)
      I[gid].parent = -1;
    const int left = I[gid].left;
    const int right = left+1;
    if (!I[gid].left_leaf) {
      I[left].parent = gid;
    }
    if (!I[gid].right_leaf) {
      I[right].parent = gid;
    }
  }
}

void BuildBinaryRadixTree_SerialKernel(__global BrtNode *I, __global BrtNode* L, __global BigUnsigned* mpoints, int mbits, int size) {
  for (int i = 0; i < size; ++i) {
    BuildBinaryRadixTree(I, L, mpoints, mbits, size, i);
  }
}
