#ifndef __OPENCL_VERSION__
  #include "../BigUnsigned/BigUnsigned.h"
  #include "../Line/Line.h"
  #include "../Vector/vec_n.h"
  #include "CellResolution.h"
#else
  #include "./OpenCL/C/BigUnsigned/BigUnsigned.h"
  #include "./OpenCL/C/Line/Line.h"
  #include "./OpenCL/C/Vector/vec_n.h"
  #include "./OpenCL/CellResolution/CellResolution.h"
#endif

/* Ambiguous cells code */
unsigned char computeOutCode(floatn point, floatn min, floatn max) {
  unsigned char mask = 0;
  if (point.x < min.x)
    mask |= 1;
  else if (point.x > max.x)
    mask |= 1 << 1;
  if (point.y < min.y)
    mask |= 1 << 2;
  else if (point.y > max.y)
    mask |= 1 << 3;
  #if DIM == 3
    if (point.z < min.z)
      mask |= 1 << 4;
    else if (point.z > max.z)
      mask |= 1 << 5;
  #endif
  return mask;
}

void sub_v2v2(cl_double2 *a, cl_double2 b, cl_double2 c) {
  a->x = b.x - c.x;
  a->y = b.y - c.y;
}
void dot_v2v2(double *dot, cl_double2 a, cl_double2 b) {
  *dot = (a.x*b.x) + (a.y*b.y);
}
void point_on_vn(cl_double2 *result, cl_double2 point, cl_double2 ray, double t) {
  result->x = point.x + ray.x * t;
  result->y = point.y + ray.y * t;
  #if DIM == 3
    result->z = point.z + ray.z * t;
  #endif
}
bool v3_on_aasquare(cl_float3 point, cl_float3 min, cl_float3 normal, float width) {
  #define CHECK_FACE(a, b, c) \
  if (normal.a != 0.0) { return ((point.b > min.b && point.b < min.b + width) && (point.c > min.c && point.c < min.c + width)); }
  CHECK_FACE(x, y, z);
  CHECK_FACE(y, z, x);
  CHECK_FACE(z, x, y);
  #undef CHECK_FACE
  return false;
}
bool v2_on_aaedge(cl_double2 point, cl_double2 min, cl_double2 normal, float width) {
  #define CHECK_EDGE(a, b) \
  if (normal.a != 0.0) { return (point.b > min.b && point.b < min.b + width); }
  CHECK_EDGE(x, y);
  CHECK_EDGE(y, x);
  #undef CHECK_EDGE
  return false;
}

//Line-box intersection tests
int doCohenSutherlandTest(floatn point1, floatn point2, floatn min, floatn max) {
  unsigned char outcode1 = computeOutCode(point1, min, max);
  unsigned char outcode2 = computeOutCode(point2, min, max);
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

static floatn getNodeCenterFromOctree(OctNode *octree, unsigned int key, unsigned int octreeSize, float octreeWidth) {
  OctNode node = octree[key];
  floatn center = { 0.0,0.0 };
  float shift = octreeWidth / (1 << (node.level)) / 2.0;

  while (node.parent != -1) {
    unsigned quadrant;
    for (quadrant = 0; quadrant < (1 << DIM); ++quadrant) {
      if (octree[node.parent].children[quadrant] == key) break;
    }

    center.x += (quadrant & (1 << 0)) ? shift : -shift;
    center.y += (quadrant & (1 << 1)) ? shift : -shift;
    key = node.parent;
    node = octree[key];
    shift *= 2.0;
  }

  return center;;
}

bool doLineBoxTest(const floatn *point1, const floatn *point2, const floatn *minimum, const floatn *maximum) {
#if DIM == 3
  cl_double3 dMinimum = { minimum->x, minimum->y, minimum->z };
  cl_double3 newPoint = {};

#else
  cl_double2 dMinimum = { minimum->x, minimum->y };
  cl_double2 dMaximum = { maximum->x, maximum->y };
  cl_double2 newPoint = {0.0,0.0};
  cl_double2 dPoint1 = { point1->x, point1->y };
  cl_double2 dPoint2 = { point2->x, point2->y };
#endif

  //Needs testing...
  int csResult = doCohenSutherlandTest(*point1, *point2, *minimum, *maximum);
  if (csResult != 0) return (csResult == 1) ? true : false;
  if (true) {
#if  DIM == 3
    cl_double3 normals[] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0,-1.0, };
#else 
    cl_double2 normals[] = { 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, };
#endif 
    cl_double2 temp = {0.0,0.0};
    double numerator, denominator;
    double width = maximum->x - minimum->x;

    //Else for each plane, check to see where the line crosses
    for (int i = 0; i < 4; ++i) {
      sub_v2v2(&temp, dMinimum, dPoint1);
      dot_v2v2(&numerator, temp, normals[i]);
      sub_v2v2(&temp, dPoint2, dPoint1);
      dot_v2v2(&denominator, temp, normals[i]);
      point_on_vn(&newPoint, dPoint1, temp, numerator / denominator);
      if (v2_on_aaedge(newPoint, dMinimum, normals[i], width)) return true;

      sub_v2v2(&temp, dMaximum, dPoint1);
      dot_v2v2(&numerator, temp, normals[i]);
      sub_v2v2(&temp, dPoint2, dPoint1);
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

int getIndexUsingBoundingBox(unsigned int key, unsigned int level, Line* lines, int* boundingBoxes, unsigned int size) {
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
cl_int FindConflictCells(OctNode *octree, unsigned int octreeSize, floatn octreeCenter, float octreeWidth,
  ConflictPair* conflictPairs, int* smallestContainingCells, unsigned int numSCCS, Line* orderedLines, unsigned int numLines, float2* points, unsigned int gid) {
  OctNode node = octree[gid];
  //If that node contains leaves...
  if (node.leaf != 0) {
    //for each leaf...
    floatn parentcenter = getNodeCenterFromOctree(octree, gid, octreeSize, octreeWidth);
    const float width = octreeWidth / (1 << (node.level + 1));
    const float shift = width / 2.0;
    for (int leafKey = 0; leafKey < 1 << DIM; ++leafKey) {
      node = octree[gid];
      if ((1 << leafKey) & node.leaf) {
        floatn center = parentcenter;
        center.x += (leafKey & (1 << 0)) ? shift : -shift;
        center.y += (leafKey & (1 << 1)) ? shift : -shift;

        const floatn minimum = { center.x - shift + octreeCenter.x, center.y - shift + octreeCenter.y };
        const floatn maximum = { center.x + shift + octreeCenter.x, center.y + shift + octreeCenter.y };

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
              if (doLineBoxTest((const floatn*)&points[orderedLines[currentIndex].firstIndex],
                (const floatn*)&points[orderedLines[currentIndex].secondIndex],
                (const floatn*)&minimum, (const floatn*)&maximum)) {
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
  return CL_SUCCESS;
}