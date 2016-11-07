#include "./SharedSources/Quantize/Quantize.h"
#include "./SharedSources/ZOrder/z_order.h"
#include "./SharedSources/Line/Line.h"
#include "./SharedSources/CellResolution/ConflictCellDetection.h"
__kernel void QuantizePointsKernel(
  __global intn *quantizePoints,
  __global floatn *points,
  const floatn minimum,
  const int reslnWidth,
  const float bbWidth
)
{
  const size_t gid = get_global_id(0);
  const floatn point = points[gid];
  quantizePoints[gid] = QuantizePoint(&point, &minimum, reslnWidth, bbWidth);
}
__kernel void PointsToMortonKernel(
  __global BigUnsigned *inputBuffer,
  __global intn *points,
  const unsigned int size,
  const unsigned int bits
  ) 
 {
 const size_t gid = get_global_id(0);
 const size_t lid = get_local_id(0);
 BigUnsigned tempBU;
 intn tempPoint = points[gid];

 if (gid < size) {
   xyz2z(&tempBU, tempPoint, bits);
 } else {
   initBlkBU(&tempBU, 0);
 }
 
 barrier(CLK_GLOBAL_MEM_FENCE);
 inputBuffer[gid] = tempBU;
}

__kernel void BitPredicateKernel( 
  __global int *inputBuffer, 
  __global unsigned *predicateBuffer, 
  unsigned index, 
  unsigned char comparedWith)
{
  BitPredicate(inputBuffer, predicateBuffer, index, comparedWith, get_global_id(0));
}

__kernel void BUBitPredicateKernel( 
  __global BigUnsigned *inputBuffer, 
  __global unsigned *predicateBuffer, 
  unsigned index, 
  unsigned char comparedWith)
{
  BUBitPredicate(inputBuffer, predicateBuffer, index, comparedWith, get_global_id(0));
}

__kernel void UniquePredicateKernel(
  __global BigUnsigned *inputBuffer,
  __global unsigned *predicateBuffer)
{
  UniquePredicate(inputBuffer, predicateBuffer, get_global_id(0));
}

__kernel void BCellPredicateKernel(
  __global BCell *inputBuffer,
  __global unsigned *predicateBuffer,
  unsigned index,
  unsigned char comparedWith,
  int mbits)
{
  BCellPredicate(inputBuffer, predicateBuffer, index, comparedWith, mbits, get_global_id(0));
}

__kernel void LevelPredicateKernel(
  __global BCell *inputBuffer,
  __global unsigned *predicateBuffer,
  unsigned index,
  unsigned char comparedWith,
  int mbits)
{
  LevelPredicate(inputBuffer, predicateBuffer, index, comparedWith, mbits, get_global_id(0));
}
__kernel void StreamScanKernel( 
  __global unsigned* buffer, 
  __global unsigned* result, 
  __global volatile int* I, 
  __local unsigned* localBuffer, 
  __local unsigned* scratch)
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
    __local unsigned *tmp = scratch;
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
  __global unsigned *lPredicateBuffer, 
  __global unsigned *leftBuffer, 
  unsigned size)
{
  BUCompact(inputBuffer, resultBuffer, lPredicateBuffer, leftBuffer, size, get_global_id(0));
}

__kernel void CompactKernel( 
  __global int *inputBuffer, 
  __global int *resultBuffer, 
  __global unsigned *lPredicateBuffer, 
  __global unsigned *leftBuffer, 
  unsigned size)
{
  Compact(inputBuffer, resultBuffer, lPredicateBuffer, leftBuffer, size, get_global_id(0));
}

//Double Compaction
__kernel void BCellFacetCompactKernel( 
  __global BCell *inputBCellBuffer, 
  __global BCell *resultBCellBuffer, 
  __global int *inputIndexBuffer, 
  __global int *resultIndexBuffer, 
  __global unsigned *lPredicateBuffer, 
  __global unsigned *leftBuffer, 
  unsigned size)
{
  BCellFacetCompact( inputBCellBuffer, inputIndexBuffer, resultBCellBuffer, resultIndexBuffer, lPredicateBuffer, leftBuffer, size, get_global_id(0));
}


//Single Compaction
__kernel void BUSingleCompactKernel(
  __global BigUnsigned *inputBuffer,
  __global BigUnsigned *resultBuffer,
  __global unsigned *predicateBuffer,
  __global unsigned *addressBuffer)
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
  if (gid < size - 1) {
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
__kernel void GetBCellLCPKernel(
  __global Line* lines,
  __global BigUnsigned* zpoints,
  __global BCell* bCells,
  __global int* facetIndices,
  const int mbits 
  ) {
  const int gid = get_global_id(0);
  GetBCellLCP(lines, zpoints, bCells, facetIndices, mbits, gid);
}

__kernel void LookUpOctnodeFromBCellKernel(
  __global BCell* bCells,
  __global OctNode* octree,
  __global int* BCellToOctree
  ) {
  const int gid = get_global_id(0);
  BigUnsigned lcp = bCells[gid].lcp;
  int lcpLength = bCells[gid].lcpLength;
  int index = getOctNode(lcp, lcpLength, octree);
  barrier(CLK_LOCAL_MEM_FENCE);
  BCellToOctree[gid] = index;
}

__kernel void FindConflictCellsKernel(
  __global OctNode *octree, 
  __global FacetPair *facetPairs,
  __global intn* qPoints,
  __global Line* lines,
  __global int* nodeToFacet,
  __global Conflict* conflicts,
  unsigned int numLines, 
  OctreeData od
) {
  const int gid = get_global_id(0);
  FindConflictCells(
          octree, facetPairs, &od, 
          conflicts, nodeToFacet, lines, numLines, qPoints, gid);
}

__kernel void CountResolutionPointsKernel(
  __global Conflict* conflicts,
  __global Line* orderedLines,
  __global intn* qPoints,
  __global int* predicates,
  __global ConflictInfo* info_array,
  __global int* resolutionCounts
  ) 
{
  const int gid = get_global_id(0);
  resolutionCounts[gid] = -1;
  Conflict c = conflicts[gid];
  int totalAdditionalPoints = 0;
  ConflictInfo info;
  info.num_samples = 0;
  info.currentNode = gid;
  if (c.color == -2)
  {
    Line firstLine = orderedLines[c.i[0]];
    Line secondLine = orderedLines[c.i[1]];
    intn q1 = qPoints[firstLine.firstIndex];
    intn q2 = qPoints[firstLine.secondIndex];
    intn r1 = qPoints[secondLine.firstIndex];
    intn r2 = qPoints[secondLine.secondIndex];
    sample_conflict_count(&info, q1, q2, r1, r2, c.origin, c.width);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  info_array[gid] = info;
  predicates[gid] = (info.num_samples > 0) ? 1 : 0; 
  resolutionCounts[gid] = info.num_samples;
  // resolutionCounts[0] = sizeof(LinePair);
}

__kernel void GetResolutionPointsKernel(
  __global Conflict* conflicts,
  __global Line* orderedLines,
  __global intn* qPoints,
  __global int* predicates,
  __global ConflictInfo* info_array,
  __global int* scannedCounts,
  __global intn* resolutionPoints
  ) 
{
  const int gid = get_global_id(0);
  Conflict c = conflicts[gid];
  ConflictInfo info = info_array[gid];
  int predicator = predicates[gid];
  int offset = (gid==0) ? 0 : scannedCounts[gid-1];
  
  if (predicator == 1)
  {
    Line firstLine = orderedLines[c.i[0]];
    Line secondLine = orderedLines[c.i[1]];
    intn q1 = qPoints[firstLine.firstIndex];
    intn q2 = qPoints[firstLine.secondIndex];
    intn r1 = qPoints[secondLine.firstIndex];
    intn r2 = qPoints[secondLine.secondIndex];

   //This is really bad in terms of efficient global memory usage... 
    const int n = info.num_samples;
    for (int i = 0; i < n; ++i) {
      floatn sample;
      sample_conflict_kernel(i, &info, &sample);
      resolutionPoints[offset + i] = convert_intn(sample);
    }
  }
}

__kernel void GetFacetPairsKernel(
  __global int* BCellToOctree, 
  __global FacetPair *facetPairs, 
  int numLines
)
{
  const int gid = get_global_id(0);
  int leftNeighbor = (gid == 0) ? -1 : BCellToOctree[gid - 1];
  int rightNeighbor = (gid == numLines-1) ? -1 : BCellToOctree[gid + 1];
  int me = BCellToOctree[gid];
  //If my left neighbor doesn't go to the same octnode I go to
  if (leftNeighbor != me) {
      //Then I am the first BCell/Facet belonging to my octnode
      facetPairs[me].first = gid;
  }
  //If my right neighbor doesn't go the the same octnode I go to
  if (rightNeighbor != me) {
      //Then I am the last BCell/Facet belonging to my octnode
      facetPairs[me].last = gid;
  }
}