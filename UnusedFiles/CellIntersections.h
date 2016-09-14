#ifndef __CELL_INTERSECTIONS_H__
#define __CELL_INTERSECTIONS_H__

#include "./opencl/defs.h"
#include "./opencl/FloatSegment.h"

// A CellIntersections object stores multiple labels for each of a
// cell's children.
class CellIntersections {
 public:
  static const int NUM_LABELS = 2;
  static const int NUM_OCTANTS = (1<<DIM);

 public:
  CellIntersections() {
    for (int i = 0; i < NUM_LABELS * NUM_OCTANTS; ++i) {
      l[i] = -1;
    }
  }
  void set(const int octant, const int label, const FloatSegment& seg) {
    for (int i = octant*NUM_LABELS; i < (octant+1)*NUM_LABELS; ++i) {
      if (l[i] == -1) {
        l[i] = label;
        segs[i] = seg;
        break;
      }
      if (l[i] == label) {
        if (seg.length2() > segs[i].length2()) {
          // Keep the longest segment for the given label
          segs[i] = seg;
        }
        break;
      }
    }
  }
  bool is_multi(const int octant) const {
    return l[octant*NUM_LABELS+1] > -1;
  }
  int num_labels(const int octant) const {
    for (int i = 0; i < NUM_LABELS; ++i) {
      if (l[octant*NUM_LABELS+i] < 0)
        return i;
    }
    return NUM_LABELS;
  }
  FloatSegment seg(const int i, const int octant) const {
    return segs[octant*NUM_LABELS+i];
  }
  int label(const int i, const int octant) const {
    return l[octant*NUM_LABELS+i];
  }
 private:
  int l[NUM_LABELS*NUM_OCTANTS];
  FloatSegment segs[NUM_LABELS*NUM_OCTANTS];
};

#endif
