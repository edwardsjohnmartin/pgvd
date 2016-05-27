#include "Karras.h"

#include <iostream>
#include <algorithm>
#include <memory>
#include <string>

#include "BoundingBox.h"
#include "CLWrapper.h"
#include "timer.h"

extern "C" {
  #include "ParallelAlgorithms.h"
	#include "BuildOctree.h"
  #include "BuildBRT.h"
}
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
  CLWrapper CL(256, 256);
  Timer t;
intn z2xyz(BigUnsigned *z, const Resln* resln) {
  intn p = make_intn(0);
	BigUnsigned temp, tempb;
	BigUnsigned zero;
	initBlkBU(&zero, 0);
  for (int i = 0; i < resln->bits; ++i) {
    for (int j = 0; j < DIM; ++j) {
      //if ((z & (BigUnsigned(1) << (i*DIM+j))) > 0)
			initBlkBU(&temp, 1);
			shiftBULeft(&tempb, &temp, i*DIM + j);
			andBU(&temp, z, &tempb);
			if (compareBU(&temp, &zero) > 0) {
				p.s[j] |= (1 << i);
			}
    }
  }
  return p;
}

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
  intn q = make_intn(0);
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
    ret.push_back(make_intn(0));
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

vector<OctNode> BuildOctreeInParallel(
    const vector<intn>& points, const Resln& resln, const bool verbose) {
  system("cls");
  if (verbose) {
    t.restart("Octree build time:");
    CL.verbose = true;
  }
  else {
    CL.verbose = false;
  }
  if (points.empty())
    throw logic_error("Zero points not supported");
  vector<OctNode> Octree;
  int n = points.size();
  int octreeSize;
  CL.UploadPoints(points);
  CL.ConvertPointsToMorton(n, resln.bits);
  CL.RadixSort(n, resln.mbits);
  CL.UniqueSorted(n);
  CL.BuildBinaryRadixTree(n, resln.mbits);
  CL.BinaryRadixToOctree(n, octreeSize);
  CL.DownloadOctree(Octree, octreeSize);

  if (verbose) {
    t.stop();
  }
  return Octree;
}

vector<OctNode> BuildOctreeInSerial(
  const vector<intn>& points, const Resln& resln, const bool verbose) {
  if (points.empty())
    throw logic_error("Zero points not supported");
  vector<OctNode> Octree;
  int n = points.size();
  int nextPowerOfTwo = max((int)pow(2, ceil(log(points.size()) / log(2))), 8);
  vector<BigUnsigned> mpoints_vec(nextPowerOfTwo);

  for (int i = 0; i < n; ++i) {
    BigUnsigned temp;
    xyz2z(&temp, points[i], resln.bits);
    mpoints_vec.push_back(temp);
  }
  for (int i = n; i < nextPowerOfTwo; ++i) {
    BigUnsigned temp;
    initBlkBU(&temp, 0);
    mpoints_vec.push_back(temp);
  }

  sort(mpoints_vec.rbegin(), mpoints_vec.rend(), lessThanBigUnsigned);
  n = unique(mpoints_vec.begin(), mpoints_vec.end(), equalsBigUnsigned) - mpoints_vec.begin();
  BigUnsigned* mpoints = mpoints_vec.data();
  
  vector<BrtNode> I_vec(n-1);
  vector<BrtNode> L_vec(n);
  BrtNode* I = I_vec.data();
  BrtNode* L = L_vec.data();
  BuildBinaryRadixTree_SerialKernel(I, L, mpoints, resln.mbits, n);

  vector<unsigned int> local_splits_vec(n);
  unsigned int* local_splits = local_splits_vec.data();
  ComputeLocalSplits_SerialKernel(local_splits, I, n);

  vector<unsigned int> prefix_sums_vec(n);
  unsigned int* prefix_sums = prefix_sums_vec.data();
  StreamScan_SerialKernel((Index*)local_splits, (Index*)prefix_sums, n);

  Octree.resize(prefix_sums[n - 1]);
  brt2octree_kernel(I, Octree.data(), local_splits, prefix_sums, n);

  return Octree;
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
