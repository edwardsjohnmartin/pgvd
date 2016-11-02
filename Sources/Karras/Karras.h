#ifndef __KARRAS_H__
#define __KARRAS_H__

#include <vector>
#include <stdexcept>

// #include "./SharedSources/OpenCL/defs.h"
#include "../../SharedSources/Vector/vec.h"
extern "C" {
  #include "../../SharedSources/OctreeResolution/Resln.h"
}
#include "../../SharedSources/BoundingBox/BoundingBox.h"
#include "../../SharedSources/ZOrder/z_order.h"
#include "../../SharedSources/Octree/OctNode.h"

namespace Karras {

// Quantize a single point.
// dwidth is passed in for performance reasons. It is equal to
//   float dwidth = bb.max_size();
intn Quantize(
    const floatn& p, const Resln& resln,
    const BoundingBox& bb, const float dwidth, const bool clamped);

std::vector<intn> Quantize(
    const floatn* points, const unsigned int numPoints, const Resln& r,
    const BoundingBox* customBB, const bool clamped = false);

std::vector<OctNode> BuildOctreeInParallel(
    const std::vector<intn>& opoints, const Resln& r, const bool verbose=false);

std::vector<OctNode> BuildOctreeInSerial(
  const std::vector<intn>& opoints, const Resln& r, const bool verbose = false);

// Debug output
// void OutputOctree(const std::vector<OctNode>& octree);
void OutputOctree(const OctNode* octree, const int n);

} // namespace

// inline std::ostream& operator<<(std::ostream& out, const Karras::Resln& resln) {
//   out << "width=" << resln.width << ", volume=" << resln.volume
//       << ", bits=" << resln.bits
//       << ", mbits=" << resln.mbits;
//   return out;
// }

#endif
