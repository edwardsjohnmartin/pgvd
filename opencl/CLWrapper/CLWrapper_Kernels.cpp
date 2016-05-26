#include "CLWrapper.h"
#include <math.h>

extern "C" {
#include "BuildOctree.h"
}
//--PRIVATE--//
void CLWrapper::initRadixSortBuffers(){
  if (!isBufferUsable(buffers.predicate, sizeof(Index)* (globalSize)))
    buffers.predicate = createBuffer(sizeof (Index)* (globalSize));
  if (!isBufferUsable(buffers.address, sizeof(Index)* (globalSize)))
    buffers.address = createBuffer(sizeof(Index)*(globalSize));
  if (!isBufferUsable(buffers.bigUnsignedResult, sizeof(BigUnsigned)* (globalSize)))
     buffers.bigUnsignedResult = createBuffer(sizeof(BigUnsigned)*globalSize);
  if (!isBufferUsable(buffers.bigUnsignedResultCopy, sizeof(BigUnsigned)* (globalSize))) {
    buffers.bigUnsignedResultCopy = createBuffer(sizeof(BigUnsigned)*globalSize);
    BigUnsigned zero;
    initBlkBU(&zero, 0);
    clEnqueueFillBuffer(queue, buffers.bigUnsignedResultCopy->getBuffer(), &zero, sizeof(BigUnsigned), 0, sizeof(BigUnsigned)* (globalSize), 0, NULL, NULL);
  }

  //get local size from kernel
  size_t streamScanLocalSize = kernelBox->getSteamScanWorkGroupSize(globalSize);
  if (!isBufferUsable(buffers.intermediate, sizeof(cl_int)*(globalSize / streamScanLocalSize)))
    buffers.intermediate = createBuffer(sizeof(cl_int)*(globalSize / streamScanLocalSize));
  if (!isBufferUsable(buffers.intermediateCopy, sizeof(cl_int)*(globalSize / streamScanLocalSize))) {
    buffers.intermediateCopy = createBuffer(sizeof(cl_int)*(globalSize / streamScanLocalSize));
    Index negativeOne = -1;
    clEnqueueFillBuffer(queue, buffers.intermediateCopy->getBuffer(), &negativeOne, sizeof(Index), 0, sizeof(Index) * globalSize/streamScanLocalSize, 0, NULL, NULL);
  }

}
void CLWrapper::initUniqueBuffers() {
  // PBuffer: 0. ABuffer: 1. Intermediate: 2 Result: 3
  if (!isBufferUsable(buffers.predicate, sizeof(Index)* (globalSize)))
    buffers.predicate = createBuffer(sizeof(Index)*(globalSize));
  if (!isBufferUsable(buffers.address, sizeof(Index)* (globalSize)))
    buffers.address = createBuffer(sizeof(Index)*(globalSize));
  size_t streamScanLocalSize = kernelBox->getSteamScanWorkGroupSize(globalSize);
  if (!isBufferUsable(buffers.intermediate, sizeof(cl_int)*(globalSize / streamScanLocalSize)))
    buffers.intermediate = createBuffer(sizeof(cl_int)*(globalSize / streamScanLocalSize));
  if (!isBufferUsable(buffers.bigUnsignedResult, sizeof(BigUnsigned)* (globalSize)))
    buffers.bigUnsignedResult = createBuffer(sizeof(Index)*(globalSize));
}
void CLWrapper::InitBinaryRadixTreeBuffers() {
  if (!isBufferUsable(buffers.internalNodes, sizeof(BrtNode)* (globalSize)))
    buffers.internalNodes = createBuffer(sizeof(BrtNode)* (globalSize));
  if (!isBufferUsable(buffers.leafNodes, sizeof(BrtNode)* (globalSize)))
    buffers.leafNodes = createBuffer(sizeof(BrtNode)* (globalSize));
}
void CLWrapper::InitBinaryRadixToOctreeBuffers(size_t n) {
  if (!isBufferUsable(buffers.localSplits, sizeof(unsigned int)* (n)))
    buffers.localSplits = createBuffer(sizeof(unsigned int)* (n));
  if (!isBufferUsable(buffers.localSplitsCopy, sizeof(unsigned int)* (n))) {
    buffers.localSplitsCopy = createBuffer(sizeof(unsigned int)* (n));
    unsigned int zero = 0;
    clEnqueueFillBuffer(queue, buffers.localSplitsCopy->getBuffer(), &zero, sizeof(unsigned int), 0, sizeof(unsigned int)* (n), 0, NULL, NULL);
  }
  if (!isBufferUsable(buffers.scannedSplits, sizeof(unsigned int)* (n)))
    buffers.scannedSplits = createBuffer(sizeof(unsigned int)* (n));
}

void CLWrapper::envokeRadixSortRoutine(const int size, const Index mbits){
  //Input: 0. LPBuffer: 1. RPBuffer: 2. LABuffer: 3. RABuffer: 4. Result: 5.
  //Using default workgroup size.
  const size_t globalWorkSize[] = { globalSize, 0, 0 };
  const size_t localWorkSize[] = { localSize, 0, 0 };
  
  shared_ptr<Buffer> temp;
  //For each bit
  for (Index index = 0; index < mbits; index++) {
    //Predicate the 0's and 1's
    kernelBox->bitPredicate(buffers.bigUnsignedInput->getBuffer(), buffers.predicate->getBuffer(), index, 1, globalSize);

    //Scan the predication buffers.
    kernelBox->streamScan(buffers.predicate->getBuffer(), buffers.intermediate->getBuffer(), buffers.intermediateCopy->getBuffer(), buffers.address->getBuffer(), globalSize);

	  //Compacting
    kernelBox->doubleCompact(buffers.bigUnsignedInput->getBuffer(), buffers.bigUnsignedResult->getBuffer(), buffers.bigUnsignedResultCopy->getBuffer(), buffers.predicate->getBuffer(), buffers.address->getBuffer(), globalSize);

    //Swap result with input.
    temp = buffers.bigUnsignedInput;
    buffers.bigUnsignedInput = buffers.bigUnsignedResult;
    buffers.bigUnsignedResult = temp;
  }
}

inline std::string buToString(BigUnsigned bu) {
  std::string representation = "";
  if (bu.len == 0)
  {
    if (bu.isNULL) {
      representation += "[NULL]";
    }
    else {
      representation += "[0]";
    }
  }
  else {
    for (int i = bu.len; i > 0; --i) {
      representation += "[" + std::to_string(bu.blk[i - 1]) + "]";
    }
  }

  return representation;
}

void CLWrapper::UploadPoints(const vector<intn>& points) {
  if (verbose) {
    timer.restart("Uploading points:");
  }

  globalSize = max((int)pow(2, ceil(log(points.size()) / log(2))), 8);
  if (!isBufferUsable(buffers.points, sizeof(intn)* (globalSize)))
    buffers.points = createBuffer(sizeof(intn)* (globalSize));

  intn* gpuPoints = (intn*)buffers.points->map_buffer();
  memcpy(gpuPoints, points.data(), points.size() * sizeof(intn));
  buffers.points->unmap_buffer();
  if (verbose) {
    clFinish(queue);
    timer.stop();
  }
}
void CLWrapper::DownloadOctree(vector<OctNode> &octree_vec, const int octree_size) {
  if (verbose) {
    timer.restart("Downloading octree:");
  }
  octree_vec.resize(octree_size);
  OctNode* tempOctree = (OctNode*)clEnqueueMapBuffer(queue, buffers.octree->getBuffer(), CL_TRUE, CL_MAP_READ, 0, sizeof(OctNode)*(octree_size), 0, NULL, NULL, NULL);
  memcpy(octree_vec.data(), tempOctree, sizeof(OctNode)*(octree_size));
  clEnqueueUnmapMemObject(queue, buffers.octree->getBuffer(), tempOctree, 0, NULL, NULL);
  if (verbose) {
    clFinish(queue);
    timer.stop();
  }
}
void CLWrapper::ConvertPointsToMorton(const int size, const int bits) {
  if (verbose) {
    timer.restart("Convering points to morton:");
  }
  globalSize = max((int)pow(2, ceil(log(size) / log(2))), 8);
  if (!isBufferUsable(buffers.bigUnsignedInput, sizeof(BigUnsigned)* (globalSize)))
    buffers.bigUnsignedInput = createBuffer(sizeof(BigUnsigned)* (globalSize));

  kernelBox->pointsToMorton(buffers.bigUnsignedInput->getBuffer(), buffers.points->getBuffer(), size, bits, globalSize);
  if (verbose) {
    clFinish(queue);
    timer.stop();
  }
}


//--WRAPPERS--//
void CLWrapper::RadixSort(const int size, const Index mBits) {
  if (verbose) {
    timer.restart("Running radix sort:");
  }
  globalSize = max((int)pow(2, ceil(log(size) / log(2))), 8); //buffers.bigUnsignedInput->getSize()/sizeof(BigUnsigned);
  localSize = min((int)globalSize, 256);//min((int)pow(2, floor(log(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE)/log(2))), (int)globalSize);
    
  initRadixSortBuffers();       
  envokeRadixSortRoutine(size, mBits);

  if (verbose) {
    clFinish(queue);
    timer.stop();
  }
};
void CLWrapper::UniqueSorted(int &newSize) {
  if (verbose) {
    timer.restart("Running unique:");
  }
  globalSize = buffers.bigUnsignedInput->getSize()/sizeof(BigUnsigned);
  
  initUniqueBuffers();
  kernelBox->uniquePredicate(buffers.bigUnsignedInput->getBuffer(), buffers.predicate->getBuffer(), globalSize);
  kernelBox->streamScan(buffers.predicate->getBuffer(), buffers.intermediate->getBuffer(), buffers.intermediateCopy->getBuffer(), buffers.address->getBuffer(), globalSize);
  
  kernelBox->singleCompact(buffers.bigUnsignedInput->getBuffer(), buffers.bigUnsignedResult->getBuffer(), buffers.predicate->getBuffer(), buffers.address->getBuffer(), globalSize);
  
  shared_ptr<Buffer> temp = buffers.bigUnsignedInput;
  buffers.bigUnsignedInput = buffers.bigUnsignedResult;
  buffers.bigUnsignedResult = temp;
  
  if (verbose) {
    clFinish(queue);
    timer.stop();
  }
  clEnqueueReadBuffer(queue, buffers.address->getBuffer(), CL_TRUE, (sizeof(Index)*globalSize-(sizeof(Index))), sizeof(Index), &newSize, 0, NULL,NULL);
}
void CLWrapper::BuildBinaryRadixTree(size_t n, int mbits) {
  if (verbose) {
    timer.restart("Running BuildBinaryRadixTree:");
  }
  globalSize = buffers.bigUnsignedInput->getSize() / sizeof(BigUnsigned);
  InitBinaryRadixTreeBuffers();
  kernelBox->buildBinaryRadixTree(buffers.internalNodes->getBuffer(), buffers.leafNodes->getBuffer(), buffers.bigUnsignedInput->getBuffer(), mbits, n, globalSize);
  if (verbose) {
    clFinish(queue);
    timer.stop();
  }
}
void CLWrapper::BinaryRadixToOctree(size_t n, int &octree_size) {
  if (verbose) {
    timer.restart("Running BinaryRadixToOctree:");
  }
  
  vector<unsigned int> local_splits_vec(n - 1, 0); // be sure to initialize to zero
  vector<unsigned int> prefix_sums_vec(n);
  int nextPowerOfTwo = max((int)pow(2, ceil(log(n) / log(2))), 8);
  InitBinaryRadixToOctreeBuffers(nextPowerOfTwo);
  
  //compute local splits
  kernelBox->computeLocalSplits(buffers.localSplits->getBuffer(), buffers.localSplitsCopy->getBuffer(), buffers.internalNodes->getBuffer(), n, nextPowerOfTwo);
  //scan the splits
  kernelBox->streamScan(buffers.localSplits->getBuffer(), buffers.intermediate->getBuffer(), buffers.intermediateCopy->getBuffer(), buffers.scannedSplits->getBuffer(), nextPowerOfTwo);

  //Read in the required octree size
  int* temp = (int*)clEnqueueMapBuffer(queue, buffers.scannedSplits->getBuffer(), CL_TRUE, CL_MAP_READ, sizeof(int)*(n - 2), sizeof(int), 0, NULL, NULL, NULL); //SLOW!!!
  octree_size = *temp;//= prefix_sums[n - 1];
  clEnqueueUnmapMemObject(queue, buffers.scannedSplits->getBuffer(), temp, 0, NULL, NULL);
  
  //Make it a power of two.
  const int nextOctreeSizePowerOfTwo = max((int)pow(2, ceil(log(octree_size) / log(2))), 8);
  
  //Create an octree buffer.
  if (!isBufferUsable(buffers.octree, sizeof(OctNode)* (nextOctreeSizePowerOfTwo))) {
    buffers.octree = createBuffer(sizeof(OctNode)* (nextOctreeSizePowerOfTwo));
  }
  //use the scanned splits & brt to create octree.
  kernelBox->brt2Octree_init(buffers.internalNodes->getBuffer(), buffers.octree->getBuffer(), buffers.localSplits->getBuffer(), buffers.scannedSplits->getBuffer(), n, nextOctreeSizePowerOfTwo);
  kernelBox->brt2Octree(buffers.internalNodes->getBuffer(), buffers.octree->getBuffer(), buffers.localSplits->getBuffer(), buffers.scannedSplits->getBuffer(), n, n);
  
  if (verbose) {
    clFinish(queue);
    timer.stop();
  }
}

