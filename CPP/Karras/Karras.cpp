#include "Karras.h"

#include <iostream>
#include <algorithm>
#include <memory>
#include <string>

#include "../../C/BoundingBox/BoundingBox.h"
#include "clfw.hpp"
#include "../Kernels/Kernels.h"
#include "../Timer/timer.h"

using std::cout;
using std::endl;
using std::vector;
using std::shared_ptr;

// kWidth is the quantized width in one dimension.
// Number of bits per dimension is kNumBits = log(kWidth).
// Total number of bits for morton code is kNumBits * DIM.
// static const int kWidth = 8;
// static const int kNumBits = 3;

namespace Karras {
  Timer t;

bool lessThanBigUnsigned(BigUnsigned& a, BigUnsigned&b) {
	if (compareBU(&a, &b) == -1) {
		return 1;
	}
	return 0;
}
bool equalsBigUnsigned(BigUnsigned& a, BigUnsigned &b) {
	if (compareBU(&a, &b) == 0) {
		return 1;
	}
	return 0;
}

// dwidth is passed in for performance reasons. It is equal to
//   float dwidth = bb.max_size();
int_n Quantize(
    const float_n& p, const Resln& resln,
    const BoundingBox& bb, const float dwidth, const bool clamped) {
  int_n q = {0,0};
  int effectiveWidth = resln.width-1;
  if (clamped) {
    effectiveWidth = resln.width;
  }
  for (int k = 0; k < DIM; ++k) {
    // const double d =
    //     (resln.width-1) * ((p.s[k] - bb.min().s[k]) / dwidth);
    const double d = effectiveWidth * ((p.s[k] - bb.minimum.s[k]) / dwidth);
    const int v = static_cast<int>(d+0.5);
    if (v < 0) {
      cerr << "Coordinate in dimension " << k << " is less than zero.  d = "
           << d << " v = " << v << endl;
      cerr << "  p[k] = " << p.s[k]
           << " bb.min()[k] = " << bb.minimum.s[k] << endl;
      cerr << "  dwidth = " << dwidth << " kwidth = " << resln.width << endl;
      throw logic_error("bad coordinate");
    }
    q.s[k] = v;
  }
  return q;
}

vector<int_n> Quantize(
    const float_n* points, const unsigned int numPoints, 
    const Resln& resln,
    const BoundingBox* bb, const bool clamped) {
  if (numPoints <0) return vector<int_n>();

  float dwidth;
  BB_max_size(bb, &dwidth);
  if (dwidth == 0) {
    vector<int_n> ret;
    ret.push_back({0,0});
    return ret;
  }

  // Quantize points to integers
  vector<int_n> qpoints(numPoints);
  for (int i = 0; i < numPoints; ++i) {
    const float_n& p = points[i];
    const int_n q = Quantize(p, resln, *bb, dwidth, clamped);
    qpoints[i] = q;
  }
  
  return qpoints;
}

inline std::string buToString(BigUnsigned bu) {
  std::string representation = "";
  if (bu.len == 0)
  {
    representation += "[0]";
  }
  else {
    for (int i = bu.len; i > 0; --i) {
      representation += "[" + std::to_string(bu.blk[i - 1]) + "]";
    }
  }
  return representation;
}

// Debug output
// void OutputOctreeNode(
//     const int node, const std::vector<OctNode>& octree, vector<int> path) {
void OutputOctreeNode(
    const int node, const OctNode* octree, vector<int> path) {
  for (int i = 0; i < 4; ++i) {
    vector<int> p(path);
    p.push_back(i);

    for (int i : p) {
      cout << i;
    }
    cout << endl;

    if (!is_leaf(&octree[node], i))
      OutputOctreeNode(octree[node][i], octree, p);
  }
}

void OutputOctree(const OctNode* octree, const int n) {
  // if (!octree.empty()) {
  if (n > 0) {
    cout << endl;
    vector<int> p;
    p.push_back(0);
    OutputOctreeNode(0, octree, p);
  }
}

} // namespace