#ifndef __OCT_CELL_H__
#define __OCT_CELL_H__

#include "./Resln.h"
#include "./BoundingBox.h"
#include "./OctNode.h"

struct OctCell {
  OctCell() : parent(0) {}
  OctCell(const intn origin_, const int width_,
          const int parent_idx_,
          OctNode const* parent_, int octant_,
          OctNode const* node_, int data_)
      : origin(origin_), width(width_),
        parent_idx(parent_idx_), parent(parent_), octant(octant_),
        node(node_), data(data_) {}

  intn get_origin() const { return origin; }
  int get_width() const { return width; }
  int get_parent_idx() const { return parent_idx; }
  OctNode const* get_parent() const { return parent; }
  int get_octant() const { return octant; }
  bool is_leaf() const { return ::is_leaf(parent, octant); }
  OctNode const* get_node() const {
    if (is_leaf()) {
      throw std::logic_error("Cannot get node from a non-leaf cell");
    }
    return node;
  }
  int get_data() const {
    if (!is_leaf()) {
      throw std::logic_error("Cannot get data from a non-leaf cell");
    }
    return (*parent)[octant];
  }
  int get_level(const Resln& resln) {
    int level = 0;
    while ((width << level) < resln.width)
      ++level;
    return level;
  }
  BoundingBox<intn> bb() const {
    return BoundingBox<intn>(origin, origin+make_uni_intn(width));
  }

  bool operator==(const OctCell& cell) const {
    return origin == cell.origin
        && width == cell.width
        && parent_idx == cell.parent_idx
        && octant == cell.octant;
  }

  friend std::ostream& operator<<(std::ostream& out, const OctCell& cell);
  friend std::istream& operator>>(std::istream& in, OctCell& cell);

 private:
  intn origin;
  int width;
  int parent_idx;
  OctNode const* parent;
  int octant;
  OctNode const* node;
  int data;
};

inline std::ostream& operator<<(std::ostream& out, const OctCell& cell) {
  out << cell.origin << " " << cell.width << " " << cell.parent_idx
      << " " << cell.octant << " " << cell.data;
  return out;
}

inline std::istream& operator>>(std::istream& in, OctCell& cell) {
  in >> cell.origin >> cell.width >> cell.parent_idx
     >> cell.octant >> cell.data;
  return in;
}


#endif
