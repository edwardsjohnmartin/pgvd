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
// const int n = info.num_samples;
// floatn* samples = new floatn[n];
// for (int i = 0; i < info.num_samples; ++i) {
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
    int num_samples;
    float s0;
    float s1;
    float alpha;
    float k1_even;
    float k2_even;
    float k1_odd;
    float k2_odd;
    floatn p_origin;
    floatn u;
} LinePair;

typedef struct ConflictInfo {
    int num_samples;
    int num_line_pairs;
    LinePair line_pairs[4];
    int offsets[4];
    int currentNode;
} ConflictInfo;

// Returns the number of samples needed to resolve a conflict in cell
// located at origin with the given width.
// int sample_conflict_count(
void sample_conflict_count(
    ConflictInfo* info,
    const intn q0, const intn q1, const intn r0, const intn r1,
    const intn origin, const int width);

void sample_conflict_kernel(const int i, ConflictInfo* info, floatn* samples);

#endif
#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#endif
