#ifdef __OPENCL_VERSION__ 
#include "opencl\C\BuildBRT.h"
#else
#include "BuildBRT.h"
#endif

#ifndef __OPENCL_VERSION__
#define __local
#define __global
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
  privateLcp.len = (mbits - length + 7) / 8;
  *value = privateValue;
  *lcp = privateLcp;
}

int compute_lcp_length(BigUnsigned* a, BigUnsigned* b, int mbits) {
  BigUnsigned tempa, tempb;
  unsigned int v = mbits; // compute the next highest power of 2 of 32-bit v
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  int offset = v >> 1;
  for (int i = v >> 2; i > 0; i >>= 1) {
    shiftBURight(&tempb, b, offset);
    shiftBURight(&tempa, a, offset);

    if (compareBU(&tempa, &tempb) == 0)
      offset -= i;
    else
      offset += i;
  }
  shiftBURight(&tempa, a, offset);
  shiftBURight(&tempb, b, offset);
  
  if (compareBU(&tempa, &tempb) == 0) {
    shiftBURight(&tempa, a, offset-1);
    shiftBURight(&tempb, b, offset-1);
    if (compareBU(&tempa, &tempb) == 0)
      return mbits - (offset - 1);
    else
      return mbits - offset;
  }
  else
    return mbits - (offset + 1);
}

void BuildBinaryRadixTree( __global BrtNode *I, __global BigUnsigned* mpoints, int mbits, int size, const unsigned int gid)
{
  BigUnsigned current;
  BigUnsigned left;
  BigUnsigned right;
  BigUnsigned temp;
  //n-1 internal nodes.
  if (gid < size-1) {

    // Determine direction of the range (+1 or -1) 
    int d;
    current = mpoints[gid];
    right = mpoints[gid + 1];
    if (gid == 0) 
      d = 1;
    else {
      left = mpoints[gid - 1];
      int l_pos = compute_lcp_length(&current, &right, mbits);
      int l_neg = compute_lcp_length(&current, &left, mbits);
      d = ((l_pos - l_neg > 0) - (l_pos - l_neg < 0)); //sign
    }

    // Compute upper bound for the length of the range
    int l;
    if (gid == 0) {
      l = size-1;
    } else {
      const int lcp_min = (d == -1) ? compute_lcp_length(&current, &right, mbits) : compute_lcp_length(&current, &left, mbits);//1ms
      int l_max = 2;
      temp = mpoints[gid + l_max*d];
      while ( gid + l_max * d >= 0 &&
              gid + l_max * d <= size - 1 && 
              compute_lcp_length( &current, &temp, mbits) > lcp_min) 
      {
        temp = mpoints[gid + l_max*d];
        l_max = l_max << 1; //Not sure if this should be before or after...
      }
      // Find the other end using binary search.
      // In some cases, the search can go right off the end of the array.
      // l_max likely is beyond the end of the array, but we need it to be
      // since it's a power of 2. So define a max length that we call l_cutoff.
      const int l_cutoff = (d==-1) ? gid : size - gid - 1;
      l = 0;
      for (int t = l_max >> 1; t >= 1; t >>= 1) { //5ms
        if (l + t <= l_cutoff) {
          temp = mpoints[gid + (l + t)*d];
          if (compute_lcp_length(&current, &temp, mbits) > lcp_min) {
            l = l + t;
          }
        }
      }
    }
    // j is the index of the other end of the range. In other words,
    // range = [i, j] or range = [j, i].
    const int j = gid + l * d;
    // Find the split position using binary search
    temp = mpoints[j];
    const int lcp_node = compute_lcp_length(&current, &temp, mbits); //1ms
    const int s_cutoff = (d==-1) ? gid - 1 : size - gid - 2;
    int s = 0;
    for (int den = 2; den < 2*l; den *= 2) { //5ms
      const int t = (int)((l + (float)den - 1) / den);// ceil(l / (float)den));
      if (s + t <= s_cutoff) {
        temp = mpoints[gid + (s + t)*d];
        if (compute_lcp_length(&current, &temp, mbits) > lcp_node) {
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
#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#endif

