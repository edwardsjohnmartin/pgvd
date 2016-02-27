#ifndef __OCTREE_UTILS_H__
#define __OCTREE_UTILS_H__

#include <vector>
#include <iostream>

#include "./opencl/defs.h"
#include "./opencl/vec.h"
#include "./OctNode.h"
#include "./OctCell.h"
extern "C" {
  #include "./Resln.h"
}

namespace OctreeUtils {

// using namespace Karras;

OctCell FindLeaf(
    const intn& p, const std::vector<OctNode>& octree, const Resln& resln);

OctCell FindNeighbor(
    const OctCell& cell, const int intersection,
    const std::vector<OctNode>& octree, const Resln& resln);

struct CellIntersection {
  CellIntersection() {}
  CellIntersection(const float t_, const floatn p_)
      : t(t_), p(p_) {}
  float t;
  floatn p;
};

// Find intersections of the line segment ab with an octree cell.
std::vector<CellIntersection> FindIntersections(
    const floatn& a, const floatn& b, const intn& origin, const int width,
    const Resln& resln);

// Find intersections of the line segment ab with an octree cell.
std::vector<CellIntersection> FindIntersections(
    // const intn& a, const intn& b, const OctCell& cell,
    const floatn& a, const floatn& b, const OctCell& cell,
    const Resln& resln);

} // namespace

#endif
