#ifndef __OPENCL_VERSION__
#include "../BigUnsigned/BigUnsigned.h"
#include "../Line/Line.h"
#include "../Vector/vec.h"
#include "ConflictCellDetection.h"
#include "../Quantize/Quantize.h"
#include "../../Sources/GLUtilities/Sketcher.h"
#define __local
#define __global
#else
#include "./SharedSources/BigUnsigned/BigUnsigned.h"
#include "./SharedSources/Line/Line.h"
#include "./SharedSources/Vector/vec.h"
#include "./SharedSources/CellResolution/ConflictCellDetection.h"
#include "./SharedSources/Quantize/Quantize.h"
#endif

bool liangBarskey(floatn *min, floatn *max, floatn *p1, floatn *p2) {
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
        if (p[i] == 0) {
            if (q[i] < 0) return false;
        }
        else {
            float d = q[i] / p[i];
            if (p[i] > 0)  tmax = (tmax <= d) ? tmax : d;
            else            tmin = (tmin >= d) ? tmin : d;
            if (tmin > tmax || tmax == 0.0 || tmin == 1.0) return false;
        }
    }
    return true;
}

static intn getNodeOrigin(__global OctNode *octree, unsigned int key, int octreeWidth) {
    OctNode node = octree[key];
    intn offset = make_intn(0, 0);
    int shift = octreeWidth / (1 << (node.level));

    while (node.parent != -1) {
        unsigned quadrant;
        for (quadrant = 0; quadrant < (1 << DIM); ++quadrant) {
            if (octree[node.parent].children[quadrant] == key) break;
        }

        offset.x += (quadrant & 1) ? shift : 0;
        offset.y += (quadrant & 2) ? shift : 0;
        shift <<= 1;
        key = node.parent;
        node = octree[node.parent];
    }

    return offset;
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
int FindConflictCells(__global OctNode *octree, OctreeData *od, __global Conflict* conflicts,
    __global int* BoundingCells, unsigned int numSCCS, __global Line* orderedLines,
    unsigned int numLines, __global intn* points, unsigned int gid) {
    OctNode start, current;
    start = current = octree[gid];

    //If that node contains leaves... //This is bad... maybe some predication/scan/compaction on leaves...
    if (start.leaf != 0) {
        floatn origin = convert_floatn(getNodeOrigin(octree, gid, od->qwidth));

//#ifndef __OPENCL_VERSION__
//        intn _o = convert_intn(origin);
//        floatn o = UnquantizePoint(&_o, &od->fmin, od->qwidth, od->fwidth);
//        using namespace GLUtilities;
//        Point p = { { o.x, o.y, 0.0, 1.0 },{ 1.0, 0.0, 0.0, 1.0 } };
//        //printf("origin for %d is %d, %d\n", gid, _o.x, _o.y);
//        GLUtilities::Sketcher::instance()->add(p);
//#endif

        const float leafWidth = od->qwidth / (1 << (start.level + 1));

        // For each potential leaf in start... 
        for (int L = 0; L < 1 << DIM; ++L) {
            if ((1 << L) & start.leaf) {
                current = start;
                floatn min_ = origin;
                min_.x += (L & 1) ? leafWidth : 0;
                min_.y += (L & 2) ? leafWidth : 0;
                floatn max_ = min_ + (leafWidth - .1F); //TODO: make liang barskey half open.

                //For each node from the current to the root
                int N = gid;
                do {
                    //Check to see if the current node is a bounding cell.
                    int firstIndex = getIndexUsingBoundingBox(N, current.level, orderedLines, BoundingCells, numLines);
                    while (firstIndex > 0 && BoundingCells[firstIndex - 1] == BoundingCells[firstIndex]) firstIndex--;

                    //If it is...
                    if (firstIndex != -1) { // && current.level != od->maxDepth - 1
                        int currentIndex = firstIndex;
                        //For each line this node is a bounding box to...
                        do {
                            intn Q1 = points[orderedLines[currentIndex].firstIndex];
                            intn Q2 = points[orderedLines[currentIndex].secondIndex];

                            //Verify the line isn't degenerate or equal
                            if (Q1.x != Q2.x || Q1.y != Q2.y) {
                                //Paint the leaves that intersect the line.
                                floatn P1 = convert_floatn(Q1); //UnquantizePoint(&Q1, &od->fmin, od->qwidth, od->fwidth); //
                                floatn P2 = convert_floatn(Q2); // UnquantizePoint(&Q2, &od->fmin, od->qwidth, od->fwidth); //

#ifndef __OPENCL_VERSION__
                                intn temp = convert_intn(P1);
                                floatn first = UnquantizePoint(&temp, &od->fmin, od->qwidth, od->fwidth);
                                temp = convert_intn(P2);
                                floatn second = UnquantizePoint(&temp, &od->fmin, od->qwidth, od->fwidth);

                                using namespace GLUtilities;
                                GLUtilities::Line l = {
                                    { { first.x, first.y, 0.0, 1.0 },{ 0.0, 1.0, 0.0, 1.0 } },
                                    { { second.x, second.y, 0.0, 1.0 },{ 0.0, 1.0, 0.0, 1.0 } } };
                                GLUtilities::Sketcher::instance()->add(l);


                                //temp = convert_intn(min_);
                                //first = UnquantizePoint(&temp, &od->fmin, od->qwidth, od->fwidth);
                                //temp = convert_intn(max_);
                                //second = UnquantizePoint(&temp, &od->fmin, od->qwidth, od->fwidth);
                                //l = {
                                //    { { first.x, first.y, 0.0, 1.0 },{ 0.0, 0.0, 1.0, 1.0 } },
                                //    { { second.x, second.y, 0.0, 1.0 },{ 0.0, 1.0, 0.0, 1.0 } } };
                                //GLUtilities::Sketcher::instance()->add(l);
#endif

                                int width = od->qwidth / (1 << (start.level + 1));
                                if (width > 1) {
                                    if (liangBarskey((float2*)(&min_), (float2*)(&max_), &P2, &P1)) {
                                        if (conflicts[4 * gid + L].color == -1) {
                                            conflicts[4 * gid + L].color = orderedLines[currentIndex].color;
                                            conflicts[4 * gid + L].i[0] = currentIndex;
                                        }
                                        else if (conflicts[4 * gid + L].color != orderedLines[currentIndex].color) {
                                            conflicts[4 * gid + L].color = -2;
                                            conflicts[4 * gid + L].i[1] = currentIndex;
                                            conflicts[4 * gid + L].width = width;
                                            conflicts[4 * gid + L].origin = convert_intn(min_);
                                        }
                                    }
                                }
                            }
                            currentIndex++;
                        } while (currentIndex < numSCCS && BoundingCells[currentIndex] == BoundingCells[firstIndex]);
                    }
                    N = current.parent;
                    if (N >= 0) current = octree[N];
                } while (N >= 0);
            }
        }
    }
    return 0;
}
#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#endif
