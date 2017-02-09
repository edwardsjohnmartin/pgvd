#pragma once 

#include "Vector/vec.h"
#include "OctreeDefinitions/defs.h"

#ifdef __cplusplus
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
    cl_int padding[2];
    // unsigned char padding[4];
} ConflictInfo;

#ifdef __cplusplus
#include <iostream>
inline std::ostream& operator<<(std::ostream& out, const LinePair& pair) {
  out << "num_samples=" <<  pair.num_samples << "; s0=" << pair.s0
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
      << " offsets = "
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
	const intn origin, const cl_int width);

void sample_conflict_kernel(const cl_int i, ConflictInfo* info, floatn* samples);

inline bool weakCompareLineInfo(LinePair *a, LinePair *b) {
	/* a0 can at times be 10^6 off... That could be a bug... */
	if ( fabs(a->a0 - b->a0 ) > .001) return false;
	if ( fabs(a->alpha - b->alpha ) > 10.0) return false;
	if ( fabs(a->k1_even - b->k1_even ) > .001) return false;
	if ( fabs(a->k1_odd - b->k1_odd ) > .001) return false;
	if ( fabs(a->k2_even - b->k2_even ) > .001) return false;
	if ( fabs(a->k2_odd - b->k2_odd ) > .001) return false;
	if (a->num_samples != b->num_samples) return false;
	if ( fabs(a->p_origin.x - b->p_origin.x ) > .001) return false;
	if ( fabs(a->p_origin.y - b->p_origin.y ) > .001) return false;
#ifdef OCT3D
	if ( fabs(a->p_origin.z - b->p_origin.z ) > .001) return false;
#endif
	if ( fabs(a->s0 - b->s0 ) > .001) return false;
	if ( fabs(a->s1 - b->s1 ) > .001) return false;
	if ( fabs(a->u.x - b->u.x ) > .001) return false;
	if ( fabs(a->u.y - b->u.y ) > .001) return false;
#ifdef OCT3D
	if ( fabs(a->u.z - b->u.z ) > .001) return false;
#endif
	return true;
}

inline bool compareConflictInfo(ConflictInfo *a, ConflictInfo *b) {
	if (!weakCompareLineInfo(&a->line_pairs[0], &b->line_pairs[0])) return false;
	if (!weakCompareLineInfo(&a->line_pairs[1], &b->line_pairs[1])) return false;
	if (!weakCompareLineInfo(&a->line_pairs[2], &b->line_pairs[2])) return false;
	if (!weakCompareLineInfo(&a->line_pairs[3], &b->line_pairs[3])) return false;

	if (a->num_line_pairs != b->num_line_pairs) return false;
	if (a->num_samples != b->num_samples) return false;
	if (a->offsets[0] != b->offsets[0]) return false;
	if (a->offsets[1] != b->offsets[1]) return false;
	if (a->offsets[2] != b->offsets[2]) return false;
	if (a->offsets[3] != b->offsets[3]) return false;
	return true;
}

/* Run for numConflicts - 1 */
inline void predPntToConflict(__global cl_int* scannedNumPtsPerConflict, __global cl_int* predication, cl_int gid) {
	cl_int address = scannedNumPtsPerConflict[gid];
	/* Note that each conflict has at least one point, meaning each element in the scan is unique.
		Therefore, we don't need to check the previous address to see if our element has been already
		predicated. */

	predication[address] = 1;
}

#ifndef OpenCL
#undef __local
#undef __global
#endif
