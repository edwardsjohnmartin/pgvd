#include "./OpenCL/C/ZOrder/z_order.h"
#include "./OpenCL/C/Line/Line.h"
#include "./OpenCL/C/CellResolution/CellResolution.h"
__kernel void PointsToMortonKernel(
  __global BigUnsigned *inputBuffer,
  __global int_n *points,
  const unsigned int size,
  const unsigned int bits
  ) 
 {
 const size_t gid = get_global_id(0);
 const size_t lid = get_local_id(0);
 BigUnsigned tempBU;
 int_n tempPoint = points[gid];

 if (gid < size) {
   xyz2z(&tempBU, tempPoint, bits);
 } else {
   initBlkBU(&tempBU, 0);
 }
 
 barrier(CLK_GLOBAL_MEM_FENCE);
 inputBuffer[gid] = tempBU;
}

__kernel void BitPredicateKernel( 
  __global BigUnsigned *inputBuffer, 
  __global Index *predicateBuffer, 
  Index index, 
  unsigned char comparedWith)
{
  BitPredicate(inputBuffer, predicateBuffer, index, comparedWith, get_global_id(0));
}

__kernel void UniquePredicateKernel(
 __global BigUnsigned *inputBuffer,
  __global Index *predicateBuffer)
{
  UniquePredicate(inputBuffer, predicateBuffer, get_global_id(0));
}

__kernel void LinePredicateKernel(
 __global Line *inputBuffer,
  __global Index *predicateBuffer,
  unsigned index,
  unsigned char comparedWith,
  int mbits)
{
  LinePredicate(inputBuffer, predicateBuffer, index, comparedWith, mbits, get_global_id(0));
}
__kernel void LevelPredicateKernel(
 __global Line *inputBuffer,
  __global Index *predicateBuffer,
  unsigned index,
  unsigned char comparedWith,
  int mbits)
{
  LevelPredicate(inputBuffer, predicateBuffer, index, comparedWith, mbits, get_global_id(0));
}

__kernel void StreamScanKernel( 
  __global Index* buffer, 
  __global Index* result, 
  __global volatile int* I, 
  __local Index* localBuffer, 
  __local Index* scratch)
{
  const size_t gid = get_global_id(0);
  const size_t lid = get_local_id(0);
  const size_t wid = get_group_id(0);
  const size_t ls = get_local_size(0);
  int sum = 0;  
  StreamScan_Init(buffer, localBuffer, scratch, gid, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = ls / 2; offset > 0; offset >>= 1) {
    AddAll(scratch, lid, offset);
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  //ADJACENT SYNCRONIZATION
  if (lid == 0 && gid != 0) {
  while (I[wid - 1] == -1);
  I[wid] = I[wid - 1] + scratch[0];
  }
  if (gid == 0) I[0] = scratch[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  result[gid] = scratch[0];

  scratch[lid] = localBuffer[lid];
  for (unsigned int i = 1; i < ls; i <<= 1) {
    HillesSteelScan(localBuffer, scratch, lid, i);
    __local Index *tmp = scratch;
    scratch = localBuffer;
    localBuffer = tmp;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  sum = localBuffer[lid];

  if (wid != 0) sum += I[wid - 1];
  result[gid] = sum; 
}

//Double Compaction
__kernel void BUCompactKernel( 
  __global BigUnsigned *inputBuffer, 
  __global BigUnsigned *resultBuffer, 
  __global Index *lPredicateBuffer, 
  __global Index *leftBuffer, 
  Index size)
{
  BUCompact(inputBuffer, resultBuffer, lPredicateBuffer, leftBuffer, size, get_global_id(0));
}

//Double Compaction
__kernel void LineCompactKernel( 
  __global Line *inputBuffer, 
  __global Line *resultBuffer, 
  __global Index *lPredicateBuffer, 
  __global Index *leftBuffer, 
  Index size)
{
  LineCompact(inputBuffer, resultBuffer, lPredicateBuffer, leftBuffer, size, get_global_id(0));
}


//Single Compaction
__kernel void BUSingleCompactKernel(
  __global BigUnsigned *inputBuffer,
  __global BigUnsigned *resultBuffer,
  __global Index *predicateBuffer,
  __global Index *addressBuffer)
{
  BUSingleCompact(inputBuffer, resultBuffer, predicateBuffer, addressBuffer, get_global_id(0));
}


//Binary Radix Tree Builder
__kernel void BuildBinaryRadixTreeKernel(
__global BrtNode *I,
__global BigUnsigned* mpoints,
int mbits,
int size
) 
{
  BuildBinaryRadixTree(I, mpoints, mbits, size, get_global_id(0));
}


__kernel void ComputeLocalSplitsKernel(
  __global unsigned int* local_splits, 
  __global BrtNode* I,
  const int size
)
{
  const size_t gid = get_global_id(0);
  if (size > 0 && gid == 0) {
    local_splits[0] = 1 + I[0].lcp_length / DIM;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (gid >= 0 && gid < size - 1) {
    ComputeLocalSplits(local_splits, I, gid);
  }
  
}

//This is dumb and will be merged with the other kernel.
__kernel void BRT2OctreeKernel_init(
  __global BrtNode *I,
  __global OctNode *octree,
  __global unsigned int *localSplits,
  __global unsigned int *prefixSums,
  const int size
) {
  const size_t gid = get_global_id(0);
  const int octreeSize = prefixSums[size-1];
  if(gid < octreeSize)
    brt2octree_init(gid, octree);    
}

__kernel void BRT2OctreeKernel(
  __global BrtNode *I,
  __global volatile OctNode *octree,
  __global unsigned int *localSplits,
  __global unsigned int *prefixSums,
  const int size
) {
  const int gid = get_global_id(0);
  const int octreeSize = prefixSums[size-1];
  octree[0].parent = -1;
  octree[0].level = 0;
  if (gid > 0 && gid < size - 1)
    brt2octree(gid, I, octree, localSplits, prefixSums, size, octreeSize);
}

__kernel void ComputeLineLCPKernel(
  __global Line* lines,
  __global BigUnsigned* zpoints,
  const int mbits 
  ) {
  const int gid = get_global_id(0);
  calculateLineLCP(lines, zpoints, mbits, gid);
}

__kernel void ComputeLineBoundingBoxesKernel(
  __global Line* lines,
  __global OctNode* octree,
  __global int* boundingBoxes
  ) {
  const int gid = get_global_id(0);
  BigUnsigned lcp = lines[gid].lcp;
  int lcpLength = lines[gid].lcpLength;
  int index = getOctNode(lcp, lcpLength, octree);
  barrier(CLK_LOCAL_MEM_FENCE);
  boundingBoxes[gid] = index;
  OctNode node = octree[boundingBoxes[gid]];
  barrier(CLK_LOCAL_MEM_FENCE);
  lines[gid].level = (short)node.level;
}

__kernel void FindConflictCellsKernel(
  __global OctNode *octree, 
  __global float_2* points,
  __global Line* orderedLines, 
  __global int* smallestContainingCells, 
  __global ConflictPair* conflictPairs, 
  unsigned int numOctNodes, 
  unsigned int numSCCS, 
  unsigned int numLines, 
  float_n octreeCenter, 
  float octreeWidth
) {
  const int gid = get_global_id(0);
  if (gid == 0)
    for (int i = 0; i < numOctNodes; ++i) {
      FindConflictCells(octree, numOctNodes, octreeCenter, octreeWidth, conflictPairs,
           smallestContainingCells, numSCCS, orderedLines, numLines, points, i);
      // for (int j = 0; j < 4; ++j) {
      //   conflictPairs[4 * i + j].i[0] = -3;
      // }
    }
}
