#pragma once

#include "BrtNode.h"
#include "BigUnsigned/LCP.h"

#ifndef OpenCL
	#define __global
	#define __local
#endif

// UTILITY FUNCTIONS
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#ifndef OpenCL
inline int sign(const int i) {
  return (i<0) ? -1 : ((i>0) ? +1 : 0);
}
#endif

inline void BuildBinaryRadixTree(
	__global BrtNode *I, 
	__global cl_int *IColors, 
	__global big* mpoints, 
	__global cl_int *pointColors, 
	int mbits, int size, bool colored, const int gid)
{
	big current;
  big left;
  big right;
  big temp;
  //n-1 internal nodes.
  if (gid < (size - 1)) {
    // Determine direction of the range (+1 or -1) 
    int d;
    current = mpoints[gid];
    right = mpoints[gid + 1];

    if (gid == 0)
      d = 1;
    else {
      left = mpoints[gid - 1];
      int l_pos = compute_lcp_length(&current, &right, mbits, gid);
      int l_neg = compute_lcp_length(&current, &left, mbits, gid);
      d = ((l_pos - l_neg > 0) - (l_pos - l_neg < 0)); //sign
    }

    // Compute upper bound for the length of the range
    int l;
    if (gid == 0) {
      l = size - 1;
    }
    else {
      int lcp_min = (d == -1) ? compute_lcp_length(&current, &right, mbits, gid) : compute_lcp_length(&current, &left, mbits, gid);//1ms

      int l_max = 1; //used to = 2... Causes seg fault when gid = 1 and dir = -1
      temp = mpoints[gid + l_max*d];
      while ((int)(gid + l_max * d) >= 0 &&
        gid + l_max * d <= size - 1 &&
        compute_lcp_length(&current, &temp, mbits, gid) > lcp_min)
      {
        temp = mpoints[gid + l_max*d];
        l_max = l_max << 1; //Not sure if this should be before or after...
      }

      // Find the other end using binary search.
      // In some cases, the search can go right off the end of the array.
      // l_max likely is beyond the end of the array, but we need it to be
      // since it's a power of 2. So define a max length that we call l_cutoff.
      const int l_cutoff = (d == -1) ? gid : size - gid - 1;
      l = 0;
      for (int t = l_max >> 1; t >= 1; t >>= 1) { //5ms
        if (l + t <= l_cutoff) {
          temp = mpoints[gid + (l + t)*d];
          if (compute_lcp_length(&current, &temp, mbits, gid) > lcp_min) {
            l = l + t;
          }
        }
      }
    }
    // j is the index of the other end of the range. In other words,
    // range = [i, j] or range = [j, i].
    int j = gid + l * d;

		// Find the split position using binary search
    temp = mpoints[j];
    int lcp_node = compute_lcp_length(&current, &temp, mbits, gid); //1ms
    int s_cutoff = (d == -1) ? gid - 1 : size - gid - 2;
    int s = 0;
    for (int den = 2; den < 2 * l; den *= 2) { //5ms
      int t = (int)((l + (float)den - 1) / den);// ceil(l / (float)den));
      if (s + t <= s_cutoff) {
        temp = mpoints[gid + (s + t)*d];
        if (compute_lcp_length(&current, &temp, mbits, gid) > lcp_node) {
          s = s + t;
        }
      }
		}					"? ? ?";

    int split = gid + s * d + MIN(d, 0); //(y in the paper)

    // Output child pointers
    I[gid].left = split;
    I[gid].left_leaf = (MIN(gid, j) == split);
    I[gid].right_leaf = (MAX(gid, j) == split + 1);
    compute_lcp_bu(&I[gid].lcp.bu, &mpoints[gid], lcp_node, mbits);
    I[gid].lcp.len = lcp_node;

    //Set parents
    if (gid == 0)
      I[gid].parent = -1;
    const int left = I[gid].left;
    const int right = left + 1;
    if (!I[gid].left_leaf) {
      I[left].parent = gid;
    }
    if (!I[gid].right_leaf) {
      I[right].parent = gid;
    }

		if (colored) {
			// Take on the left color if it exists.
			cl_int brtColor = (MIN(gid, j) == split) ? pointColors[split] : -1;
			
			// Find the right color if it exists.
			cl_int rightColor = (MAX(gid, j) == split + 1) ? pointColors[split + 1] : -1;

			// If the left doesn't have a color, take the right one.
			if (brtColor == -1) brtColor = rightColor;

			// Else if both left and right leaves have mismatching colors, mark as required.
			else if (rightColor != -1 && brtColor != rightColor) brtColor = -2;
			IColors[gid] = brtColor;
		}
  }
}

//inline void BuildColoredBinaryRadixTree(__global BrtNode *I, __global BigUnsigned* mpoints, __global cl_int *pointColors, __global cl_int brtColors, int mbits, int size, const unsigned int gid)
//{
//	BigUnsigned current;
//	BigUnsigned left;
//	BigUnsigned right;
//	BigUnsigned temp;
//	//n-1 internal nodes.
//	if (gid < size - 1) {
//		// Determine direction of the range (+1 or -1) 
//		int d;
//		current = mpoints[gid];
//		right = mpoints[gid + 1];
//		if (gid == 0)
//			d = 1;
//		else {
//			left = mpoints[gid - 1];
//			int l_pos = compute_lcp_length(&current, &right, mbits);
//			int l_neg = compute_lcp_length(&current, &left, mbits);
//			d = ((l_pos - l_neg > 0) - (l_pos - l_neg < 0)); //sign
//		}
//
//		// Compute upper bound for the length of the range
//		int l;
//		if (gid == 0) {
//			l = size - 1;
//		}
//		else {
//			int lcp_min = (d == -1) ? compute_lcp_length(&current, &right, mbits) : compute_lcp_length(&current, &left, mbits);//1ms
//			int l_max = 1; //used to = 2... Causes seg fault when gid = 1 and dir = -1
//			temp = mpoints[gid + l_max*d];
//			while ((int)(gid + l_max * d) >= 0 &&
//				gid + l_max * d <= size - 1 &&
//				compute_lcp_length(&current, &temp, mbits) > lcp_min)
//			{
//				temp = mpoints[gid + l_max*d];
//				l_max = l_max << 1; //Not sure if this should be before or after...
//			}
//			// Find the other end using binary search.
//			// In some cases, the search can go right off the end of the array.
//			// l_max likely is beyond the end of the array, but we need it to be
//			// since it's a power of 2. So define a max length that we call l_cutoff.
//			const int l_cutoff = (d == -1) ? gid : size - gid - 1;
//			l = 0;
//			for (int t = l_max >> 1; t >= 1; t >>= 1) { //5ms
//				if (l + t <= l_cutoff) {
//					temp = mpoints[gid + (l + t)*d];
//					if (compute_lcp_length(&current, &temp, mbits) > lcp_min) {
//						l = l + t;
//					}
//				}
//			}
//		}
//		// j is the index of the other end of the range. In other words,
//		// range = [i, j] or range = [j, i].
//		int j = gid + l * d;
//		// Find the split position using binary search
//		temp = mpoints[j];
//		int lcp_node = compute_lcp_length(&current, &temp, mbits); //1ms
//		int s_cutoff = (d == -1) ? gid - 1 : size - gid - 2;
//		int s = 0;
//		for (int den = 2; den < 2 * l; den *= 2) { //5ms
//			int t = (int)((l + (float)den - 1) / den);// ceil(l / (float)den));
//			if (s + t <= s_cutoff) {
//				temp = mpoints[gid + (s + t)*d];
//				if (compute_lcp_length(&current, &temp, mbits) > lcp_node) {
//					s = s + t;
//				}
//			}
//		}
//
//		int split = gid + s * d + MIN(d, 0);
//
//		// Output child pointers
//		I[gid].left = split;
//		I[gid].left_leaf = (MIN(gid, j) == split);
//		I[gid].right_leaf = (MAX(gid, j) == split + 1);
//		compute_lcp_bu(&I[gid].lcp.bu, &mpoints[gid], lcp_node, mbits);
//		I[gid].lcp.len = lcp_node;
//
//		//Set parents
//		if (gid == 0)
//			I[gid].parent = -1;
//		const int left = I[gid].left;
//		const int right = left + 1;
//		if (!I[gid].left_leaf) {
//			I[left].parent = gid;
//		}
//		if (!I[gid].right_leaf) {
//			I[right].parent = gid;
//		}
//	}
//}

inline void ColorBrt(__global BrtNode* brt, __global cl_int *leafColors, __global cl_int *brtColors, cl_int i) {
	//BrtNode node = brt[i];
	//if (node.left_leaf || node.right_leaf) {
	//	if (node.left_leaf && node.right_leaf) {
	//		if (leafColors[node.left])
	//	}
	//
	//	if (currentColor == -1) return;
	//	else do {

	//	} while (i != 0);
	//}
}
#undef MAX
#undef MIN
#ifndef OpenCL
	#undef __local
	#undef __global
#endif
