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
                       // const vector<Morton>& _mpoints, const Resln* _resln) {
                       const Morton* _mpoints, const Resln* _resln) {
  // if (i < 0 || i >= _mpoints.size() || 
  //     j < 0 || j >= _mpoints.size()) {
  //   throw logic_error("Illegal indices into lcp_length");
  // }
  return compute_lcp_length(_mpoints[i], _mpoints[j], _resln);
}

struct BrtNode {
  // // If lcp is 10011 and DIM == 2 then the last bit is dropped
  // // and return is [10, 01] = [2, 1] where the two values
  // // correspond to 10011 and 10011.
  // //               ^^          ^^
  // vector<int> oct_nodes() const {
  //   static const int mask = (DIM == 2) ? 3 : 7;
  //   if (DIM > 3)
  //     throw logic_error("BrtNode::oct_nodes not yet supported for D>3");
  //   const int bias = lcp_length % DIM;
  //   const int n = lcp_length / DIM;
  //   vector<int> ret(n);
  //   for (int i = 0; i < n; ++i) {
  //     const int offset = DIM * (n-i-1) + bias;
  //     // TODO: could be a bug here
  //     // ret[i] = (lcp >> offset).getBlock(0) & mask;
  //     ret[i] = (lcp >> offset) & mask;
  //   }
  //   return ret;
  // }

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

// If lcp is 10011 and DIM == 2 then the last bit is dropped
// and return is [10, 01] = [2, 1] where the two values
// correspond to 10011 and 10011.
//               ^^          ^^
// n is the number of splits to do
// vector<int> oct_nodes(const BrtNode* brt_node, const int n_) {
// vector<int> lcp2quads(const BrtNode* brt_node, const int n_, int* quadrants) {
void lcp2quads(const BrtNode* brt_node, const int n_, int* quadrants) {
  static const int mask = (DIM == 2) ? 3 : 7;
  if (DIM > 3)
    throw logic_error("BrtNode::oct_nodes not yet supported for D>3");
  const int bias = brt_node->lcp_length % DIM;
  const int n = n_;//brt_node->lcp_length / DIM;
  // vector<int> ret(n);
  // Start by shifting by, in the above example, 3. Then shift by 1.
  const int start = bias + (n-1) * DIM;
  // for (int i = 0; i < n; ++i) {
  //   const int rshift = DIM * (n-i-1) + bias;
  //   const int idx = i;
  int idx = 0;
  for (int rshift = start; rshift >= 0; rshift -= DIM, idx++) {
    // ret[idx] = (brt_node->lcp >> rshift) & mask;
    quadrants[idx] = (brt_node->lcp >> rshift) & mask;
  }
  // return ret;
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
  
  int n = points.size();
  vector<Morton> mpoints_vec(n);
  Morton* mpoints = mpoints_vec.data();
  // Morton* mpoints = new Morton[n];
  for (int i = 0; i < points.size(); ++i) {
    mpoints[i] = xyz2z(points[i], &resln);
  }

  // sort(mpoints.begin(), mpoints.end());
  sort(mpoints, mpoints + n);

  if (verbose) {
    cout << "mpoints: ";
    for (int i = 0; i < n; ++i) {
      cout << mpoints[i] << " ";
    }
    cout << endl;
  }

  // Make sure points are unique
  // std::vector<Morton>::iterator it;
  // it = std::unique(mpoints.begin(), mpoints.end());
  // mpoints.resize(std::distance(mpoints.begin(),it));

  // Morton* unique_points = new Morton[n];
  vector<Morton> unique_points_vec(n);
  unique_points_vec[0] = mpoints[0];
  int idx = 1;
  for (int i = 1; i < n; ++i) {
    if (mpoints[i] != mpoints[i-1]) {
      unique_points_vec[idx++] = mpoints[i];
    }
  }
  // delete [] mpoints;
  mpoints = unique_points_vec.data();
  n = idx;

  // // Send mpoints to gpu
  // if (o.gpu) {
  //   oct::Gpu gpu;
  //   gpu.CreateMPoints(mpoints.size());
  // }

  // const int n = mpoints.size();
  // const LcpLength lcp_length(mpoints, resln);
  // vector<BrtNode> I(n-1);
  // vector<BrtNode> L(n);
  vector<BrtNode> I_vec(n-1);
  vector<BrtNode> L_vec(n);
  BrtNode* I = I_vec.data();
  BrtNode* L = L_vec.data();
  for (int i = 0; i < n-1; ++i) {
    // Determine direction of the range (+1 or -1)
    // const int l_pos = lcp_length(i, i+1);
    // const int l_neg = lcp_length(i, i-1);
    // const int d = (i==0) ? 1 : sign(l_pos - l_neg);
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
      // const int lcp_min = lcp_length(i, i-d);
      const int lcp_min = compute_lcp_length(i, i-d, mpoints, &resln);
      int l_max = 2;
      while (i+l_max*d >= 0 && i+l_max*d <= n-1 &&
             // lcp_length(i, i+l_max*d) > lcp_min) {
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
          // if (lcp_length(i, i+(l+t)*d) > lcp_min) {
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
    // const int lcp_node = lcp_length(i, j);
    const int lcp_node = compute_lcp_length(i, j, mpoints, &resln);

    const int s_cutoff = (d==-1) ? i - 1 : n - i - 2;
    int s = 0;
    for (int den = 2; den < 2*l; den *= 2) {
      const int t = static_cast<int>(ceil(l/(float)den));
      if (s + t <= s_cutoff) {
        // if (lcp_length(i, i+(s+t)*d) > lcp_node) {
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
      // vector<int> onodes = I[i].oct_nodes();
      // vector<int> onodes = oct_nodes(&I[i], local_splits[i]);
      int quadrants[MAX_OCTREE_DEPTH];
      // vector<int> onodes = lcp2quads(&I[i], local_splits[i], quadrants);
      lcp2quads(&I[i], local_splits[i], quadrants);
      // for (const int onode : onodes) {
        // cout << onode << " ";
      for (int j = 0; j < local_splits[i]; ++j) {
        cout << quadrants[j] << " ";
      }
      cout << ") lcp_length = " << I[i].lcp_length
           << " parent = " << I[i].parent
           << endl;
    }
  }

  // Determine number of octree nodes necessary
  // First pass - initialize temporary array
  // local_splits stores how many times the octree needs to be
  // split from a parent to a child. For example, in 2D if a child
  // has an lcp_length of 8 and the parent has lcp_length of 4, then
  // the child represents two octree splits.
  // vector<int> local_splits_vec(n-1, 0); // be sure to initialize to zero
  // int* local_splits = local_splits_vec.data();
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
  // vector<int> prefix_sums(local_splits.size()+1);
  // vector<int> prefix_sums(n);
  vector<int> prefix_sums_vec(n);
  int* prefix_sums = prefix_sums_vec.data();
  // std::copy(local_splits.begin(), local_splits.end(), prefix_sums.begin()+1);
  prefix_sums[0] = 0;
  for (int i = 1; i < n; ++i) {
    prefix_sums[i] = local_splits[i-1];
  }
  // prefix_sums[0] = 0;
  // for (int i = 1; i < prefix_sums.size(); ++i) {
  for (int i = 1; i < n; ++i) {
    prefix_sums[i] = prefix_sums[i-1] + prefix_sums[i];
  }

  if (verbose) {
    cout << "Local splits: ";
    // for (int i = 0; i < local_splits.size(); ++i) {
    for (int i = 0; i < n-1; ++i) {
      cout << local_splits[i] << " ";
    }
    cout << endl;
    cout << "Prefix sums: ";
    // for (int i = 0; i < prefix_sums.size(); ++i) {
    for (int i = 0; i < n; ++i) {
      cout << prefix_sums[i] << " ";
    }
    cout << endl;
  }

  // const int splits = prefix_sums.back();
  const int splits = prefix_sums[n-1];
  if (verbose) {
    cout << "# octree splits = " << splits << endl;
  }

  // Set parent for each octree node
  vector<OctNode> octree_vec(splits);
  OctNode* octree = octree_vec.data();
  int quadrants[MAX_OCTREE_DEPTH];
  for (int brt_i = 1; brt_i < n-1; ++brt_i) {
    if (local_splits[brt_i] > 0) {
      // m = number of local splits
      const int m = local_splits[brt_i];
      // Given an lcp, lcp2quads() computes the indices
      // of the octree nodes that are created by the split
      // of this internal node. For example, if the lcp is
      // 1011, then there are two octree node splits (in 2D).
      // One child, 10, and one grandchild, 10|11 are created.
      // In this case, onodes = [10, 11].
      // const vector<int> quadrants_ = lcp2quads(&I[brt_i], m, quadrants);
      lcp2quads(&I[brt_i], m, quadrants);
      // Current octree node index
      int oct_i = prefix_sums[brt_i];
      for (int j = 0; j < m-1; ++j) {
        const int oct_parent = oct_i+1;
        // const int onode = quadrants[quadrants.size() - j - 1];
        // const int onode = quadrants[quadrants.size() - j - 1];
        const int onode = quadrants[m - j - 1];
        octree[oct_parent].set_child(onode, oct_i);
        oct_i = oct_parent;
      }
      int brt_parent = I[brt_i].parent;
      while (local_splits[brt_parent] == 0) {
        brt_parent = I[brt_parent].parent;
      }
      const int oct_parent = prefix_sums[brt_parent];
      // if (brt_parent < 0 || brt_parent >= prefix_sums.size()) {
      if (brt_parent < 0 || brt_parent >= n) {
        throw logic_error("error 0");
      }
      // if (oct_parent < 0 || oct_parent >= octree.size()) {
      if (oct_parent < 0 || oct_parent >= splits) {
        throw logic_error("error 1");
      }
      // octree[oct_parent].set_child(quadrants[quadrants.size() - m], oct_i);
      octree[oct_parent].set_child(quadrants[0], oct_i);
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

  // delete [] mpoints;

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
