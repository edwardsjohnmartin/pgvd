#ifndef __OPENCL_VERSION__
#include "../BigUnsigned/BigUnsigned.h"
#include "../Line/Line.h"
#include "../Vector/vec_n.h"
#include "CellResolution.h"
#define __local
#define __global
#else
#include "./OpenCL/C/BigUnsigned/BigUnsigned.h"
#include "./OpenCL/C/Line/Line.h"
#include "./OpenCL/C/Vector/vec_n.h"
#include "./OpenCL/C/CellResolution/CellResolution.h"
#endif

/* Ambiguous cells code */
unsigned char computeOutCode(float_n point, float_n min, float_n max) {
	unsigned char mask = 0;
	if (X_(point) < X_(min))
		mask |= 1;
	else if (X_(point) > X_(max))
		mask |= 1 << 1;
	if (Y_(point) < Y_(min))
		mask |= 1 << 2;
	else if (Y_(point) > Y_(max))
		mask |= 1 << 3;
#if DIM == 3
	if (point.z < min.z)
		mask |= 1 << 4;
	else if (point.z > max.z)
		mask |= 1 << 5;
#endif
	return mask;
}

void sub_v2v2(double_2 *a, double_2 b, double_2 c) {
	a->s[0] = X_(b) - X_(c);
	a->s[1] = Y_(b) - Y_(c);
}
void dot_v2v2(double *dot, double_2 a, double_2 b) {
	*dot = (X_(a)*X_(b)) + (Y_(a)*Y_(b));
}
void point_on_vn(double_2 *result, double_2 point, double_2 ray, double t) {
	result->s[0] = X_(point) + X_(ray) * t;
	result->s[1] = Y_(point) + Y_(ray) * t;
#if DIM == 3
	result->s[2] = Z_(point) + Z_(ray) * t;
#endif
}
bool v3_on_aasquare(float_3 point, float_3 min, float_3 normal, float width) {
#define CHECK_FACE(a, b, c) \
  if (normal.a != 0.0) { return ((point.b > min.b && point.b < min.b + width) && (point.c > min.c && point.c < min.c + width)); }
	CHECK_FACE(s[0], s[1], s[2]);
	CHECK_FACE(s[1], s[2], s[0]);
	CHECK_FACE(s[2], s[0], s[1]);
#undef CHECK_FACE
	return false;
}
bool v2_on_aaedge(double_2 point, double_2 min, double_2 normal, float width) {
#define CHECK_EDGE(a, b) \
  if (normal.a != 0.0) { return (point.b > min.b && point.b < min.b + width); }
	CHECK_EDGE(s[0], s[1]);
	CHECK_EDGE(s[1], s[0]);
#undef CHECK_EDGE
	return false;
}

//Line-box intersection tests
int doCohenSutherlandTest(float_n point1, float_n point_2, float_n min, float_n max) {
	unsigned char outcode1 = computeOutCode(point1, min, max);
	unsigned char outcode2 = computeOutCode(point_2, min, max);
	/* one of the two points is in the center*/
	if (outcode1 == 0 || outcode2 == 0)
		return 1;
	/* Both points have a side in common */
	else if (outcode1 & outcode2)
		return -1;
	/* The points are accross the cube. */
	else if ((outcode1 == 1 && outcode2 == 2) || (outcode2 == 1 && outcode1 == 2)) {
		return 1;
	}
	else if ((outcode1 == 8 && outcode2 == 4) || (outcode2 == 8 && outcode1 == 4)) {
		return 1;
	}
	else if ((outcode1 == 32 && outcode2 == 16) || (outcode2 == 32 && outcode1 == 16)) {
		return 1;
	}
	/* This is an ambiguous case */
	return 0;
}

static float_n getNodeCenterFromOctree(__global OctNode *octree, unsigned int key, unsigned int octreeSize, float octreeWidth) {
	OctNode node = octree[key];
	float_n center;
	center.s[0] = center.s[1] = 0.0;
	float shift = octreeWidth / (1 << (node.level)) / 2.0;

	while (node.parent != -1) {
		unsigned quadrant;
		for (quadrant = 0; quadrant < (1 << DIM); ++quadrant) {
			if (octree[node.parent].children[quadrant] == key) break;
		}

		X_(center) += (quadrant & (1 << 0)) ? shift : -shift;
		Y_(center) += (quadrant & (1 << 1)) ? shift : -shift;
		key = node.parent;
		node = octree[key];
		shift *= 2.0;
	}

	return center;;
}

bool doLineBoxTest(float_n *point1, float_n *point_2, float_n *minimum, float_n *maximum) {
#if DIM == 3
	double3 dMinimum = { minimum->x, minimum->y, minimum->z };
	double3 newPoint = {};

#else
	double_2 dMinimum;
	X_(dMinimum) = minimum->s[0];
	Y_(dMinimum) = minimum->s[1];
	double_2 dMaximum;
	X_(dMaximum) = maximum->s[0];
	Y_(dMaximum) = maximum->s[1];
	double_2 newPoint;
	newPoint.s[0] = newPoint.s[1] = 0.0;
	double_2 dPoint1;
	X_(dPoint1) = point1->s[0];
	Y_(dPoint1) = point1->s[1];
	double_2 dPoint_2;
	X_(dPoint_2) = point_2->s[0];
	Y_(dPoint_2) = point_2->s[1];
#endif

	//Needs testing...
	int csResult = doCohenSutherlandTest(*point1, *point_2, *minimum, *maximum);
	if (csResult != 0) return (csResult == 1) ? true : false;
	if (true) {
#if  DIM == 3
		double3 normals[] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0,-1.0, };
#else 
		double_2 normals[4];
		normals[0].s[0] =  1.0;
		normals[0].s[0] =  0.0;
		normals[1].s[0] =  0.0;
		normals[1].s[0] =  1.0;
		normals[2].s[0] = -1.0;
		normals[2].s[0] =  0.0;
		normals[3].s[0] =  0.0;
		normals[3].s[0] = -1.0;
#endif 
		double_2 temp;
		temp.s[0] = temp.s[1] = 0.0;
		double numerator, denominator;
		double width = maximum->s[0] - minimum->s[0];

		//Else for each plane, check to see where the line crosses
		for (int i = 0; i < 4; ++i) {
			sub_v2v2(&temp, dMinimum, dPoint1);
			dot_v2v2(&numerator, temp, normals[i]);
			sub_v2v2(&temp, dPoint_2, dPoint1);
			dot_v2v2(&denominator, temp, normals[i]);
			point_on_vn(&newPoint, dPoint1, temp, numerator / denominator);
			if (v2_on_aaedge(newPoint, dMinimum, normals[i], width)) return true;

			sub_v2v2(&temp, dMaximum, dPoint1);
			dot_v2v2(&numerator, temp, normals[i]);
			sub_v2v2(&temp, dPoint_2, dPoint1);
			dot_v2v2(&denominator, temp, normals[i]);
			point_on_vn(&newPoint, dPoint1, temp, numerator / denominator);
			if (v2_on_aaedge(newPoint, dMinimum, normals[i], width)) return true;

		}
	}
	return false;
}

int compareLevelThenIndex(unsigned int a_indx, unsigned int b_indx, unsigned int a_lvl, unsigned int b_lvl) {
	if (a_lvl < b_lvl)
		return -1;
	if (b_lvl < a_lvl)
		return 1;
	if (a_indx < b_indx)
		return -1;
	if (b_indx < a_indx)
		return 1;
	return 0;
}

int getIndexUsingBoundingBox(unsigned int key, unsigned int level, __global Line* lines, __global int* boundingBoxes, unsigned int size) {
	unsigned int start = 0;
	unsigned int end = size;
	while (start != end) {
		unsigned int index = ((end - start) / 2) + start;
		int x = compareLevelThenIndex(key, boundingBoxes[index], level, (lines[index].lcpLength - lines[index].lcpLength%DIM) / DIM); //lines[index].lcpLength = 3, should be 0....
		if (0 > x) end = index;
		else if (x > 0) start = index + 1;
		else return index;
	}
	return -1;
}

//Colors should be initialized as -1. run for each internal node.
int FindConflictCells(__global OctNode *octree, unsigned int octreeSize, float_n octreeCenter, float octreeWidth,
	__global ConflictPair* conflictPairs, __global int* smallestContainingCells, unsigned int numSCCS, __global Line* orderedLines, 
	unsigned int numLines, __global float_2* points, unsigned int gid) {
	OctNode node = octree[gid];
	//If that node contains leaves...
	if (node.leaf != 0) {
		//for each leaf...
		float_n parentcenter = getNodeCenterFromOctree(octree, gid, octreeSize, octreeWidth);
		const float width = octreeWidth / (1 << (node.level + 1));
		const float shift = width / 2.0;
		for (int leafKey = 0; leafKey < 1 << DIM; ++leafKey) {
			node = octree[gid];
			if ((1 << leafKey) & node.leaf) {
				float_n center = parentcenter;
				X_(center) += (leafKey & (1 << 0)) ? shift : -shift;
				Y_(center) += (leafKey & (1 << 1)) ? shift : -shift;

				float_n minimum;
				X_(minimum) = X_(center) - shift + X_(octreeCenter);
				Y_(minimum) = Y_(center) - shift + Y_(octreeCenter);

				float_n maximum;
				X_(maximum) = X_(center) + shift + X_(octreeCenter);
				Y_(maximum) = Y_(center) + shift + Y_(octreeCenter);

				//For each node from the current to the root
				int nodeKey = gid;
				do {
					//Check to see if The current node is a bounding box.
					int firstIndex = getIndexUsingBoundingBox(nodeKey, node.level, orderedLines, smallestContainingCells, numLines);
					while (firstIndex > 0 && smallestContainingCells[firstIndex - 1] == smallestContainingCells[firstIndex]) firstIndex--;

					//If it is a bounding box...
					if (firstIndex != -1) {
						unsigned int currentIndex = firstIndex;
						//For each line this node is a bounding box to...
						do {
							//Paint the leaves that intersect the line.
							float_2 firstPoint = points[orderedLines[currentIndex].firstIndex];
							float_2 secondPoint = points[orderedLines[currentIndex].secondIndex];

							if (doLineBoxTest((float_n*)&firstPoint, (float_n*)&secondPoint,
								(float_n*)&minimum, (float_n*)&maximum)) {
								if (conflictPairs[4 * gid + leafKey].i[0] == -1) {
									conflictPairs[4 * gid + leafKey].i[0] = orderedLines[currentIndex].color;
								}
								else if (conflictPairs[4 * gid + leafKey].i[0] != orderedLines[currentIndex].color) {
									conflictPairs[4 * gid + leafKey].i[0] = -2;
								}
							}
							currentIndex++;
						} while (currentIndex < numSCCS && smallestContainingCells[currentIndex] == smallestContainingCells[firstIndex]);
					}
					nodeKey = node.parent;
					if (nodeKey >= 0) node = octree[nodeKey];
				} while (nodeKey >= 0);
			}
		}
	}
	return 0;
}
#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#endif
