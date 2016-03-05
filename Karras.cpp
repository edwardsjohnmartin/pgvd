#include "./Karras.h"

#include <iostream>
#include <algorithm>
#include <memory>
#include <string>

#include "./BoundingBox.h"
// #include "./gpu.h"

extern "C" {
	#include "./C/BuildOctree.h"
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

BigUnsigned* xyz2z(BigUnsigned *result, intn p, const Resln* resln) {
  initBlkBU(result, 0);
	BigUnsigned temp;
  initBlkBU(&temp, 0);
	BigUnsigned tempb;
  initBlkBU(&tempb, 0);

	for (int i = 0; i < resln->bits; ++i) {
    for (int j = 0; j < DIM; ++j) {
			if (p.s[j] & (1 << i)) {
				//ret |= BigUnsigned(1) << (i*DIM + j);
				initBlkBU(&temp, 1);
				shiftBULeft(&tempb, &temp, i*DIM + j);
				initBUBU(&temp, result);
				orBU(result, &temp, &tempb);
			}
    }
  }
  return result;
}

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

vector<OctNode> BuildOctree(
    const vector<intn>& points, const Resln& resln, const bool verbose) {

  printBUSize();

  if (points.empty())
    throw logic_error("Zero points not supported");
  
  cout << "BigUnsigned = " << sizeof(BigUnsigned) << endl;

  int n = points.size();
  vector<BigUnsigned> mpoints_vec(n);
  BigUnsigned* mpoints = mpoints_vec.data();
  for (int i = 0; i < points.size(); ++i) {
    xyz2z(&mpoints[i], points[i], &resln);
  }

  // GPU CANDIDATE
  // Currently uses std::sort. The call below that is commented out
  // calls the (unimplemented) sort in BuildOctree.c.
  sort(mpoints, mpoints + n, lessThanBigUnsigned);
  //sort_points(mpoints, mpoints + n);

  
  // Make sure points are unique
  
	//vector<BigUnsigned> unique_points_vec(n);
  //n = unique_points(mpoints, unique_points_vec.data(), n);
	//mpoints = unique_points_vec.data();
	n = unique(mpoints, mpoints + n, equalsBigUnsigned)-(mpoints);

	if (verbose) {
		cout << "mpoints: ";
		for (int i = 0; i < n; ++i) {
			cout << buToString(mpoints[i]) << " " << endl;
		}
		cout << endl;
	}

	cout << sizeof(BigUnsigned) << endl;
  // // Send mpoints to gpu
  // if (o.gpu) {
  //   oct::Gpu gpu;
  //   gpu.CreateMPoints(mpoints.size());
  // }

  // Internal nodes
  vector<BrtNode> I_vec(n-1);
  // Leaf nodes
  vector<BrtNode> L_vec(n);
  BrtNode* I = I_vec.data();
  BrtNode* L = L_vec.data();
  build_brt(I, L, mpoints, n, &resln);

  // Set parents
  set_brt_parents(I, n);

  // local_splits stores how many times the octree needs to be
  // split from a parent to a child in the brt. For example, in 2D if a child
  // has an lcp_length of 8 and the parent has lcp_length of 4, then
  // the child represents two octree splits.
  vector<int> local_splits_vec(n-1, 0); // be sure to initialize to zero
  int* local_splits = local_splits_vec.data();

  // Determine number of octree nodes necessary
  // First pass - initialize temporary array
  compute_local_splits(local_splits, I, n);

  // Second pass - calculate prefix sums
  vector<int> prefix_sums_vec(n);
  int* prefix_sums = prefix_sums_vec.data();
  compute_prefix_sums(local_splits, prefix_sums, n);

  const int octree_size = prefix_sums[n-1];

  // Set parent for each octree node
  vector<OctNode> octree_vec(octree_size);
  OctNode* octree = octree_vec.data();
  brt2octree(I, octree, local_splits, prefix_sums, n);

  //--------------
  // Debug output
  //--------------
  if (verbose) {
    cout << endl;
    for (int i = 0; i < n-1; ++i) {
      cout << i << ": left = " << I[i].left << (I[i].left_leaf ? "L" : "I")
           << " right = " << I[i].left+1 << (I[i].right_leaf ? "L" : "I")
           << " lcp = " << buToString(I[i].lcp)
           << " oct nodes: (";
      const BrtNode* brt_node = &I[i];
      cout << ") lcp_length = " << I[i].lcp_length
           << " parent = " << I[i].parent
           << endl;
    }

    cout << "Local splits: ";
    for (int i = 0; i < n-1; ++i) {
      cout << local_splits[i] << " ";
    }
    cout << endl;
    cout << "Prefix sums: ";
    for (int i = 0; i < n; ++i) {
      cout << prefix_sums[i] << " ";
    }
    cout << endl;

    cout << "# octree splits = " << octree_size << endl;

    for (int i = 0; i < octree_size; ++i) {
      cout << i << ": ";
      for (int j = 0; j < 4; ++j) {
        cout << octree[i][j] << " ";
      }
      cout << endl;
    }
    OutputOctree(octree, octree_size);
  }

  return octree_vec;
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

// void OutputOctree(const std::vector<OctNode>& octree) {
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
