#ifndef OpenCL
// include necessary to compile on Mac
#include "GLUtilities/gl_utils.h"
#endif

#include "BigUnsigned/BigUnsigned.h"
#include "Line/Line.h"
#include "Vector/vec.h"
#include "CellResolution/ConflictCellDetection.h"
#include "Quantize/Quantize.h"
#ifndef OpenCL
#include "GLUtilities/Sketcher.h"
#define __local
#define __global
#endif
bool liangBarskey(floatn *min, floatn *max, floatn *p1, floatn *p2, int gid, int debug) {
	float p[4], q[4];
	p[0] = -(p2->x - p1->x);
	p[1] = -(p2->y - p1->y);
	p[2] = p2->x - p1->x;
	p[3] = p2->y - p1->y;

	q[0] = p1->x - min->x;
	q[1] = p1->y - min->y;
	q[2] = max->x - p1->x;
	q[3] = max->y - p1->y;

	float tmin = 0.0;
	float tmax = 1.0;
	for (int i = 0; i < 4; ++i) {
		if (p[i] == 0.0) {
			if (q[i] < 0.0) return false;
		}
		else {
			float d = q[i] / p[i];


			if (p[i] > 0.0)  tmax = (tmax <= d) ? tmax : d;
			else            tmin = (tmin >= d) ? tmin : d;
			if (tmin > tmax || tmax == 0.0 || fabs(tmin - 1.0 ) < EPSILON) return false;
		}
	}
	return true;
}

static floatn getLeafOrigin(__global OctNode *octree, OctNode parent, Leaf leaf, int octreeWidth) {
	cl_int key = leaf.parent;
	OctNode node = parent;
	floatn offset = make_floatn(0.0, 0.0);
	cl_int shift = octreeWidth / (1 << (node.level));
	cl_int leafWidth = shift / 2.0;

	while (node.parent != -1) {
		offset.x += (node.quadrant & 1) ? shift : 0;
		offset.y += (node.quadrant & 2) ? shift : 0;
		shift <<= 1;
		node = octree[node.parent];
	}

	offset.x += (leaf.quadrant & 1) ? leafWidth : 0;
	offset.y += (leaf.quadrant & 2) ? leafWidth : 0;

	return offset;
}

//Colors should be initialized as -1. run for each internal node.
void FindConflictCells(
	cl_int gid,
	__global OctNode *octree,
	__global Leaf *leaves,
	__global int* bCellToLineIndx,
	__global Pair *bCellBounds,
	__global Line* lines,
	cl_int numLines,
	cl_int keepCollisions,
	__global intn* qpoints,
	cl_int qwidth,
	__global Conflict* conflicts
	)
{
	Leaf leaf = leaves[gid];
	OctNode node = octree[leaf.parent];
	Conflict currentConflict = {};
	currentConflict.color = -1;

	//Calculate that leaf's origin/min/max.
	floatn origin = getLeafOrigin(octree, node, leaf, qwidth);
	const int leafWidth = qwidth / (1 << (node.level + 1));

	// Subtract .1F to force liang barskey to act half open.
	floatn max_ = origin + (leafWidth - .1F);

	//If the cell is resolvable...
	if (leafWidth > 1 || keepCollisions)
	{
		//...then, for each node from the leaf's parent to the root...
		int N = leaf.parent;
		while (N != -1)
		{
			//...if the current node is a bounding cell...
			Pair bounds = bCellBounds[N];
			N = node.parent;
			if (N >= 0) node = octree[N];
			if (bounds.first != -1)
			{
				//...then for each line this node bounds...
				for (int i = bounds.first; i <= bounds.last; ++i) //.001
				{
					cl_int lineIndx = bCellToLineIndx[i];
					Line line = lines[lineIndx]; // ~.002
					floatn P1 = convert_floatn(qpoints[line.first]);
					floatn P2 = convert_floatn(qpoints[line.second]); //.004

					
					//...if the line isn't degenerate...
					//if ((fabs(P1.x - P2.x) > EPSILON || fabs(P1.y - P2.y) > EPSILON))
					if ((fabs(P1.x - P2.x) > .001 || fabs(P1.y - P2.y) > .001))
					{
						//...and if the line touches the current leaf
						if (liangBarskey((float2*)(&origin), (float2*)(&max_), &P2, &P1, gid, lineIndx))
						{
							//...then if the current leaf has no color, color it with the line.
							if (currentConflict.color == -1)
							{
								currentConflict.color = line.color;
								currentConflict.q1[0] = line.first;
								currentConflict.q1[1] = line.second;
							}
							//... or if the leaf's color doesn't match the intersecting line's color...
							else if (currentConflict.color != lines[lineIndx].color)
							{
								// ... then we have a conflict cell.
								currentConflict.color = -2;
								currentConflict.q2[0] = line.first;
								currentConflict.q2[1] = line.second;
								currentConflict.width = leafWidth;
								currentConflict.origin = convert_intn(origin);
							}
						}
					}
				}
			}
		}
	}
	conflicts[gid] = currentConflict;
}

bool compareConflict(Conflict *a, Conflict *b) {
	bool test = true;
	test &= (a->color == b->color);
	test &= (a->q1[0] == b->q1[0]);
	test &= (a->q1[1] == b->q1[1]);
	test &= (a->q2[0] == b->q2[0]);
	test &= (a->q2[1] == b->q2[1]);
	test &= (a->origin.x == b->origin.x);
	test &= (a->origin.y == b->origin.y);
	test &= (a->width == b->width);
	return test;
}

void CompactConflicts(__global Conflict *inputBuffer, __global Conflict *resultBuffer, __global cl_int *predicationBuffer,
	__global cl_int *addressBuffer, cl_int size, const cl_int gid)
{
	cl_int a = addressBuffer[gid];
	cl_int b = addressBuffer[size - 2];
	cl_int c = predicationBuffer[gid];
	cl_int e = predicationBuffer[size - 1];

	//Check out http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html figure 39-14
	cl_int t = gid - a + (e + b);
	cl_int d = (!c) ? t : a - 1;

#ifdef OpenCL
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif

	resultBuffer[d] = inputBuffer[gid];
}

#ifndef OpenCL
#undef __local
#undef __global
#endif
