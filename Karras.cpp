#include "Karras.h"

#include <iostream>
#include <algorithm>
#include <memory>
#include <string>

#include "BoundingBox.h"
#include "clfw.hpp"
#include "Kernels.h"
#include "timer.h"

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
intn Quantize(
    const floatn& p, const Resln& resln,
    const BoundingBox<floatn>& bb, const float dwidth, const bool clamped) {
  intn q = {0,0};
  int effectiveWidth = resln.width-1;
  if (clamped) {
    effectiveWidth = resln.width;
  }
  for (int k = 0; k < DIM; ++k) {
    // const double d =
    //     (resln.width-1) * ((p.s[k] - bb.min().s[k]) / dwidth);
    const double d =
        effectiveWidth * ((p.s[k] - bb.min().s[k]) / dwidth);
    const int v = static_cast<int>(d+0.5);
    if (v < 0) {
      cerr << "Coordinate in dimension " << k << " is less than zero.  d = "
           << d << " v = " << v << endl;
      cerr << "  p[k] = " << p.s[k]
           << " bb.min()[k] = " << bb.min().s[k] << endl;
      cerr << "  dwidth = " << dwidth << " kwidth = " << resln.width << endl;
      throw logic_error("bad coordinate");
    }
    q.s[k] = v;
  }
  return q;
}

vector<intn> Quantize(
    const vector<floatn>& points, const Resln& resln,
    const BoundingBox<floatn>* customBB, const bool clamped) {
  if (points.empty())
    return vector<intn>();

  BoundingBox<floatn> bb;
  if (customBB) {
    bb = *customBB;
  } else {
    for (const floatn& p : points) {
      bb(p);
    }
  }
  const float dwidth = bb.max_size();
  if (dwidth == 0) {
    vector<intn> ret;
    ret.push_back({0,0});
    return ret;
  }

  // Quantize points to integers
  vector<intn> qpoints(points.size());
  for (int i = 0; i < points.size(); ++i) {
    const floatn& p = points[i];
    const intn q = Quantize(p, resln, bb, dwidth, clamped);
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

//vector<OctNode> BuildOctreeInParallel(
//    const vector<intn>& points, const Resln& resln, const bool verbose) {
//  system("cls");
//  if (verbose) {
//    t.restart("Octree build time:");
//    CLFW::verbose = true;
//  }
//  else {
//    CLFW::verbose = false;
//  }
//  if (points.empty())
//    throw logic_error("Zero points not supported");
//  vector<OctNode> Octree;
//  int n = points.size();
//  int octreeSize;
//  CL.UploadPoints(points);
//  CL.ConvertPointsToMorton(n, resln.bits);
//  CL.RadixSort(n, resln.mbits);
//  CL.UniqueSorted(n);
//  CL.BuildBinaryRadixTree(n, resln.mbits);
//  CL.BinaryRadixToOctree(n, octreeSize);
//  CL.DownloadOctree(Octree, octreeSize);
//
//  if (verbose) {
//    t.stop();
//  }
//  return Octree;
//}

vector<OctNode> BuildOctreeInSerial( const vector<intn>& points, const Resln& resln, const bool verbose) {
  if (points.empty())
    throw logic_error("Zero points not supported");
  int numPoints = points.size();
  int roundNumPoints = Kernels::nextPow2(points.size());
  vector<BigUnsigned> zpoints(roundNumPoints);

  //Points to Z Order
  Kernels::PointsToMorton_s(points.size(), resln.bits, (cl_int2*)points.data(), zpoints.data());

  //Sort and unique Z points
  sort(zpoints.rbegin(), zpoints.rend(), weakCompareBU);
  numPoints = unique(zpoints.begin(), zpoints.end(), weakEqualsBU) - zpoints.begin();

  //Build BRT
  vector<BrtNode> I(numPoints-1);
  Kernels::BuildBinaryRadixTree_s(zpoints.data(), I.data(), numPoints, resln.mbits);

  //Build Octree
  vector<OctNode> octree;
  Kernels::BinaryRadixToOctree_s(I, octree, numPoints);
  return octree;
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
