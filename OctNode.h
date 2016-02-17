#ifndef __OCT_NODE_H__
#define __OCT_NODE_H__

// An octree node is an internal node of the octree. An octree cell
// is a general term that refers to both internal nodes and leaves.

static const int leaf_masks[] = { 1, 2, 4, 8 };

struct OctNode {
 public:
  OctNode() : leaf(15) {
    std::fill(children, children + (1<<DIM), -1);
  }
  bool is_leaf(const int i) const {
    return leaf & leaf_masks[i];
  }
  void set_child(const int octant, const int child) {
    children[octant] = child;
    if (child > -1) {
      leaf &= ~leaf_masks[octant];
    } else {
      leaf |= leaf_masks[octant];
    }
  }
  void set_data(const int octant, const int data) {
    if (!is_leaf(octant))
      throw std::logic_error("Trying to set data on a non-leaf cell");
    children[octant] = data;
  }
  const int& operator[](const int i) const {
    return children[i];
  }

  friend std::ostream& operator<<(std::ostream& out, const OctNode& node);
  friend std::istream& operator>>(std::istream& in, OctNode& node);

 private:
  int children[1<<DIM];
  unsigned char leaf;
};

inline std::ostream& operator<<(std::ostream& out, const OctNode& node) {
  for (int i = 0; i < 1<<DIM; ++i) {
    out << node.children[i] << " ";
  }
  out << static_cast<int>(node.leaf);
  return out;
}

inline std::istream& operator>>(std::istream& in, OctNode& node) {
  for (int i = 0; i < 1<<DIM; ++i) {
    in >> node.children[i];
  }
  int leaf;
  in >> leaf;
  node.leaf = static_cast<unsigned char>(leaf);
  return in;
}

inline std::ostream& operator<<(
    std::ostream& out, const std::vector<OctNode>& octree) {
  out << octree.size() << " ";
  for (const OctNode& node : octree) {
    out << node << " ";
  }
  return out;
}

inline std::istream& operator>>(
    std::istream& in, std::vector<OctNode>& octree) {
  int n;
  in >> n;
  octree.resize(n);
  for (int i = 0; i < n; ++i) {
    in >> octree[i];
  }
  return in;
}


#endif
