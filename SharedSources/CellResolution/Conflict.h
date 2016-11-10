#ifndef __CONFLICT_H__
#define __CONFLICT_H__

#ifdef __OPENCL_VERSION__
	#include "./SharedSources/Vector/vec.h"
	#include "./SharedSources/OctreeDefinitions/defs.h"
#elif defined __cplusplus
	#include "../Vector/vec.h"
	#include "../OctreeDefinitions/defs.h"

//I think defs.h might handle __local and __global...
	#define __local
	#define __global
#endif

//------------------------------------------------------------
// Usage:
//
// ConflictInfo info;
// sample_conflict_count(
//     &info, q0, q1, r0, r1, origin, width, NULL, NULL);
// const cl_int n = info.num_samples;
// floatn* samples = new floatn[n];
// for (cl_int i = 0; i < info.num_samples; ++i) {
//   sample_conflict_kernel(i, &info, samples);
// }
//
// q0 and q1 are the endpoints of one line segment; r0 and r1
// are the endpoints of the other line segment. origin is the
// coordinate of the lower-left corner of the octree cell,
// and width is the width of the octree cell.
//
// After the above code has executed the sample points are
// in the samples variable.
//------------------------------------------------------------

typedef struct LinePair {
  cl_int num_samples;
  cl_float s0;
  cl_float s1;
  cl_float alpha;
  cl_float k1_even;
  cl_float k2_even;
  cl_float k1_odd;
  cl_float k2_odd;
  floatn p_origin;
  floatn u;
  cl_float a0;
  // unsigned char padding[4];
  cl_float padding;
} LinePair;

typedef struct ConflictInfo {
    cl_int num_samples;
    cl_int num_line_pairs;
    LinePair line_pairs[4];
    cl_int offsets[4];
    cl_int currentNode;
    // unsigned char padding[4];
    cl_float padding;
} ConflictInfo;

#ifdef __cplusplus
#include <iostream>
inline std::ostream& operator<<(std::ostream& out, const LinePair& pair) {
  out << "n=" <<  pair.num_samples << "; s0=" << pair.s0
      << "; s1=" << pair.s1 << "; alpha="
      << pair.alpha << "; k1_even="
      << pair.k1_even << "; k2_even=" << pair.k2_even << "; k1_odd="
      << pair.k1_odd << "; k2_odd=" << pair.k2_odd << "; a0="
      <<  pair.a0 << "; "
      << "";
  return out;
}
inline std::ostream& operator<<(std::ostream& out, const ConflictInfo& info) {
  out << "" <<  info.num_samples << " " << info.num_line_pairs
      << " " << info.currentNode << " offsets = "
      << info.offsets[0] << " " <<  info.offsets[1] << " "
      << info.offsets[2] << " " <<  info.offsets[3] << " "
      << " padding " << info.padding
      << " line_pairs[0] " << info.line_pairs[0] << "";
  if (info.num_line_pairs > 4) {
    out << " ****************";
  }
  return out;
}
inline bool operator==(const ConflictInfo& a, const ConflictInfo& b) {
  return a.num_samples == b.num_samples &&
      a.num_line_pairs == b.num_line_pairs &&
      a.currentNode == b.currentNode &&
      a.offsets[0] == b.offsets[0] &&
      a.offsets[1] == b.offsets[1] &&
      a.offsets[2] == b.offsets[2] &&
      a.offsets[3] == b.offsets[3];
}
inline bool operator!=(const ConflictInfo& a, const ConflictInfo& b) {
  return !(a == b);
}
#endif

// Returns the number of samples needed to resolve a conflict in cell
// located at origin with the given width.
// cl_int sample_conflict_count(
void sample_conflict_count(
    ConflictInfo* info,
    const intn q0, const intn q1, const intn r0, const intn r1,
    const intn origin, const int width);

void sample_conflict_kernel(const cl_int i, ConflictInfo* info, floatn* samples);

#endif
#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#endif
