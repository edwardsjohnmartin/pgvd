#include "./Karras.h"

#include <iostream>
#include <algorithm>
#include <memory>

#include "./BoundingBox.h"
// #include "./gpu.h"

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

Morton xyz2z(intn p, const Resln* resln) {
  Morton ret = 0;
  for (int i = 0; i < resln->bits; ++i) {
    for (int j = 0; j < DIM; ++j) {
      if (p.s[j] & (1<<i))
        ret |= Morton(1) << (i*DIM+j);
    }
  }
  return ret;
}

intn z2xyz(const Morton z, const Resln* resln) {
  intn p = make_intn(0);
  for (int i = 0; i < resln->bits; ++i) {
    for (int j = 0; j < DIM; ++j) {
      if ((z & (Morton(1) << (i*DIM+j))) > 0)
        p.s[j] |= (1<<i);
    }
  }
  return p;
}

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
Morton compute_lcp(const Morton value, const int length, const Resln* resln) {
  Morton mask(0);
  for (int i = 0; i < length; ++i) {
    mask |= (Morton(1) << (resln->mbits - 1 - i));
  }
  const Morton lcp = (value & mask) >> (resln->mbits - length);
  return lcp;
}

// Longest common prefix, denoted \delta in karras2014
int compute_lcp_length(const Morton a, const Morton b, const Resln* resln) {
  for (int i = resln->mbits-1; i >= 0; --i) {
    const Morton mask = Morton(1) << i;
    if ((a & mask) != (b & mask)) {
      return resln->mbits - i - 1;
    }
  }
  return resln->mbits;
}

int compute_lcp_length(const int i, const int j,
                       const Morton* _mpoints, const Resln* _resln) {
  return compute_lcp_length(_mpoints[i], _mpoints[j], _resln);
}

struct BrtNode {
  // left child (right child = left+1)
  int left;
  // Whether the left (resp. right) child is a leaf or not
  bool left_leaf, right_leaf;
  // The longest common prefix
  Morton lcp;
  // Number of bits in the longest common prefix
  int lcp_length;

  // Secondary - computed in a second pass
  int parent;
};

// Given a lcp, returns the i'th quadrant starting from the most local.
// Suppose node.lcp is 1010011 and DIM == 2. The
// quadrantInLcp(node, 0) returns 01 (1010011)
//                                        ^^
// quadrantInLcp(node, 1) returns 10 (1010011)
//                                      ^^
// quadrantInLcp(node, 2) returns 10 (1010011)
//                                    ^^
int quadrantInLcp(const BrtNode* brt_node, const int i) {
  static const int mask = (DIM == 2) ? 3 : 7;
  if (DIM > 3)
    throw logic_error("BrtNode::oct_nodes not yet supported for D>3");
  const int rem = brt_node->lcp_length % DIM;
  const int rshift = i * DIM + rem;
  return (brt_node->lcp >> rshift) & mask;
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
  if (points.empty())
    throw logic_error("Zero points not supported");
  
  cout << "Morton = " << sizeof(Morton) << endl;

  int n = points.size();
  vector<Morton> mpoints_vec(n);
  Morton* mpoints = mpoints_vec.data();
  for (int i = 0; i < points.size(); ++i) {
    mpoints[i] = xyz2z(points[i], &resln);
  }

  sort(mpoints, mpoints + n);

  if (verbose) {
    cout << "mpoints: ";
    for (int i = 0; i < n; ++i) {
      cout << mpoints[i] << " ";
    }
    cout << endl;
  }

  // Make sure points are unique
  vector<Morton> unique_points_vec(n);
  unique_points_vec[0] = mpoints[0];
  int idx = 1;
  for (int i = 1; i < n; ++i) {
    if (mpoints[i] != mpoints[i-1]) {
      unique_points_vec[idx++] = mpoints[i];
    }
  }
  mpoints = unique_points_vec.data();
  n = idx;

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
  for (int i = 0; i < n-1; ++i) {
    // Determine direction of the range (+1 or -1)
    int d;
    if (i == 0) {
      d = 1;
    } else {
      const int l_pos = compute_lcp_length(i, i+1, mpoints, &resln);
      const int l_neg = compute_lcp_length(i, i-1, mpoints, &resln);
      d = sign(l_pos - l_neg);
    }
    // Compute upper bound for the length of the range
    int l;
    if (i == 0) {
      l = n-1;
    } else {
      const int lcp_min = compute_lcp_length(i, i-d, mpoints, &resln);
      int l_max = 2;
      while (i+l_max*d >= 0 && i+l_max*d <= n-1 &&
             compute_lcp_length(i, i+l_max*d, mpoints, &resln) > lcp_min) {
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
          if (compute_lcp_length(i, i+(l+t)*d, mpoints, &resln) > lcp_min) {
            l = l + t;
          }
        }
      }
    }

    // j is the index of the other end of the range. In other words,
    // range = [i, j] or range = [j, i].
    const int j = i + l * d;
    // Find the split position using binary search
    const int lcp_node = compute_lcp_length(i, j, mpoints, &resln);

    const int s_cutoff = (d==-1) ? i - 1 : n - i - 2;
    int s = 0;
    for (int den = 2; den < 2*l; den *= 2) {
      const int t = static_cast<int>(ceil(l/(float)den));
      if (s + t <= s_cutoff) {
        if (compute_lcp_length(i, i+(s+t)*d, mpoints, &resln) > lcp_node) {
          s = s + t;
        }
      }
    }
    const int split = i + s * d + min(d, 0);
    // Output child pointers
    I[i].left = split;
    I[i].left_leaf = (min(i, j) == split);
    I[i].right_leaf = (max(i, j) == split+1);
    I[i].lcp = compute_lcp(mpoints[i], lcp_node, &resln);
    I[i].lcp_length = lcp_node;
  }

  // Set parents
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

  // local_splits stores how many times the octree needs to be
  // split from a parent to a child. For example, in 2D if a child
  // has an lcp_length of 8 and the parent has lcp_length of 4, then
  // the child represents two octree splits.
  vector<int> local_splits_vec(n-1, 0); // be sure to initialize to zero
  int* local_splits = local_splits_vec.data();

  // Debug output
  if (verbose) {
    cout << endl;
    for (int i = 0; i < n-1; ++i) {
      cout << i << ": left = " << I[i].left << (I[i].left_leaf ? "L" : "I")
           << " right = " << I[i].left+1 << (I[i].right_leaf ? "L" : "I")
           << " lcp = " << I[i].lcp
           << " oct nodes: (";
      const BrtNode* brt_node = &I[i];
      for (int j = 0; j < local_splits[i]; ++j) {
        cout << quadrantInLcp(brt_node, j) << " ";
      }
      cout << ") lcp_length = " << I[i].lcp_length
           << " parent = " << I[i].parent
           << endl;
    }
  }

  // Determine number of octree nodes necessary
  // First pass - initialize temporary array
  if (n > 0)
    local_splits[0] = 1 + I[0].lcp_length / DIM;
  for (int i = 0; i < n-1; ++i) {
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
  // Second pass - calculate prefix sums
  vector<int> prefix_sums_vec(n);
  int* prefix_sums = prefix_sums_vec.data();
  prefix_sums[0] = 0;
  for (int i = 1; i < n; ++i) {
    prefix_sums[i] = local_splits[i-1];
  }
  for (int i = 1; i < n; ++i) {
    prefix_sums[i] = prefix_sums[i-1] + prefix_sums[i];
  }

  if (verbose) {
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
  }

  const int splits = prefix_sums[n-1];
  if (verbose) {
    cout << "# octree splits = " << splits << endl;
  }

  // Set parent for each octree node
  vector<OctNode> octree_vec(splits);
  OctNode* octree = octree_vec.data();
  for (int brt_i = 1; brt_i < n-1; ++brt_i) {
    if (local_splits[brt_i] > 0) {
      // m = number of local splits
      const int m = local_splits[brt_i];
      const BrtNode* brt_node = &I[brt_i];
      // Current octree node index
      int oct_i = prefix_sums[brt_i];
      for (int j = 0; j < m-1; ++j) {
        const int oct_parent = oct_i+1;
        const int onode = quadrantInLcp(brt_node, j);
        octree[oct_parent].set_child(onode, oct_i);
        oct_i = oct_parent;
      }
      int brt_parent = I[brt_i].parent;
      while (local_splits[brt_parent] == 0) {
        brt_parent = I[brt_parent].parent;
      }
      const int oct_parent = prefix_sums[brt_parent];
      if (brt_parent < 0 || brt_parent >= n) {
        throw logic_error("error 0");
      }
      if (oct_parent < 0 || oct_parent >= splits) {
        throw logic_error("error 1");
      }
      octree[oct_parent].set_child(quadrantInLcp(brt_node, m-1), oct_i);
    }
  }

  if (verbose) {
    for (int i = 0; i < splits; ++i) {
      cout << i << ": ";
      for (int j = 0; j < 4; ++j) {
        cout << octree[i][j] << " ";
      }
      cout << endl;
    }
    OutputOctree(octree, splits);
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

    if (!octree[node].is_leaf(i))
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
