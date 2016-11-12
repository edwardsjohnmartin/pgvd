#ifndef __OPENCL_VERSION__
#include "../../Sources/GLUtilities/Sketcher.h"
#include "../BigUnsigned/BigUnsigned.h"
#include "../Line/Line.h"
#include "../Vector/vec.h"
#include "ConflictCellDetection.h"
#include "../Quantize/Quantize.h"
#define __local
#define __global
#else
#include "./SharedSources/OctreeDefinitions/defs.h"
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

//Colors should be initialized as -1. run for each internal node.
void FindConflictCells(
    __global OctNode *octree,
    __global FacetPair *facetPairs,
    OctreeData *od,
    __global Conflict* conflicts,
    __global int* nodeToFacet,
    __global Line* lines,
    unsigned int numLines,
    __global intn* points,
    unsigned int gid)
{
    OctNode start, current;
    start = current = octree[gid];

    //If the start contains leaves... //This is bad...
    if (start.leaf != 0) 
    {
        floatn origin = convert_floatn(getNodeOrigin(octree, gid, od->qwidth));
        const float leafWidth = od->qwidth / (1 << (start.level + 1));

        // For each leaf in start... 
        for (int L = 0; L < 1 << DIM; ++L) 
        {
            if ((1 << L) & start.leaf) 
            {
                //calculate that leaf's origin/min/max.
                current = start;
                floatn min_ = origin;
                min_.x += (L & 1) ? leafWidth : 0;
                min_.y += (L & 2) ? leafWidth : 0;
                // Note: subtracting .1F forces liang barskey to act half open.
                floatn max_ = min_ + (leafWidth - .1F); 
                intn Q1, Q2, Q3, Q4;
                intn Q5 = make_intn(-1, -1);
                intn Q6 = make_intn(-1, -1);
                
                //Then, for each node from the leaf's parent to the root
                int N = gid;
                while (N != -1) 
                {
                    //If the current node is a bCell...
                    int firstIndex = facetPairs[N].first;
                    int lastIndex = facetPairs[N].last;
                    if (firstIndex != -1) 
                    {
                        //Then for each line this node is a bCell to...
                        for (int currentIndex = firstIndex; currentIndex <= lastIndex; ++currentIndex) 
                        {
                            Q1 = points[lines[nodeToFacet[currentIndex]].firstIndex];
                            Q2 = points[lines[nodeToFacet[currentIndex]].secondIndex];

                            //If the line isn't degenerate...
                            if (Q1.x != Q2.x || Q1.y != Q2.y) 
                            {
                                //Get the points that define the line.
                                floatn P1 = convert_floatn(Q1);
                                floatn P2 = convert_floatn(Q2);

                                //If the cell is resolvable...
                                int width = od->qwidth / (1 << (start.level + 1));
                                if (width > 1) 
                                {
                                    //If the line touches cell
                                    if (liangBarskey((float2*)(&min_), (float2*)(&max_), &P2, &P1)) 
                                    {
                                        //If the cell has no color, then color it.
                                        if (conflicts[4 * gid + L].color == -1) 
                                        {
                                            //Keep track of the line to later look for a closer one.
                                            Q3 = make_intn( Q1.x, Q1.y );
                                            Q4 = make_intn( Q2.x, Q2.y );
                                            conflicts[4 * gid + L].color = lines[nodeToFacet[currentIndex]].color;
                                            conflicts[4 * gid + L].i[0] = nodeToFacet[currentIndex];
                                        }
                                        //Else if the colors don't match, then the cell is a conflict cell.
                                        else if (conflicts[4 * gid + L].color != lines[nodeToFacet[currentIndex]].color)
                                        {
                                            //If there is no closer conflicting line... 
                                           // if (Q5.x != -1 || (distSegmentToSegment(Q3, Q4, Q5, Q6) > distSegmentToSegment(Q3, Q4, Q1, Q2)))
                                            //{
                                                Q5 = make_intn(Q1.x, Q1.y);
                                                Q6 = make_intn(Q2.x, Q2.y);
                                                conflicts[4 * gid + L].color = -2;
                                                conflicts[4 * gid + L].i[1] = nodeToFacet[currentIndex];
                                                conflicts[4 * gid + L].width = width;
                                                conflicts[4 * gid + L].origin = convert_intn(min_);
                                            //}
                                        }
                                    }
                                }
                            }
                        };
                    }
                    N = current.parent;
                    if (N >= 0) current = octree[N];
                }
            }
        }
    }
}
#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#endif
