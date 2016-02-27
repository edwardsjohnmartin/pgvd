#ifndef __KARRAS_H__
#define __KARRAS_H__

#include <vector>
#include <stdexcept>

// #include "./opencl/defs.h"
#include "./opencl/vec.h"
extern "C" {
  #include "./Resln.h"
}
#include "./OctNode.h"
#include "./BoundingBox.h"

namespace Karras {

Morton* xyz2z(BigUnsigned *result, intn p, const Resln* resln);
intn z2xyz(Morton *z, const Resln* resln);

// Quantize a single point.
// dwidth is passed in for performance reasons. It is equal to
//   float dwidth = bb.max_size();
intn Quantize(
    const floatn& p, const Resln& resln,
    const BoundingBox<floatn>& bb, const float dwidth, const bool clamped);

std::vector<intn> Quantize(
    const std::vector<floatn>& points, const Resln& r,
    const BoundingBox<floatn>* customBB = 0, const bool clamped = false);

std::vector<OctNode> BuildOctree(
    const std::vector<intn>& opoints, const Resln& r, const bool verbose=false);

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
