#pragma once
#include "KernelBox_.h"
#include "BufferInitializers.h"

namespace KernelBox {
  cl_int PointsToMorton_p(cl::Buffer &points, cl::Buffer &zpoints, cl_int size, cl_int bits) {
    cl_int error = 0;
    size_t globalSize = nextPow2(size);
    error |= CLFW::get(zpoints, "zpoints", globalSize * sizeof(BigUnsigned));
    cl::Kernel kernel = CLFW::Kernels["PointsToMortonKernel"];
    error |= kernel.setArg(0, zpoints);
    error |= kernel.setArg(1, points);
    error |= kernel.setArg(2, size);
    error |= kernel.setArg(3, bits);
    error |= CLFW::DefaultQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(nextPow2(size)), cl::NullRange);
    return error;
  };
  cl_int PointsToMorton_s(cl_int size, cl_int bits, cl_int2* points, BigUnsigned* result) {
    int nextPowerOfTwo = nextPow2(size);
    for (int gid = 0; gid < nextPowerOfTwo; ++gid) {
      if (gid < size) {
        xyz2z(&result[gid], points[gid], bits);
      }
      else {
        initBlkBU(&result[gid], 0);
      }
    }
    return 0;
  }

  cl_int BitPredicate(cl::Buffer &input, cl::Buffer &predicate, unsigned int &index, unsigned char compared, cl_int globalSize) {
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["BitPredicateKernel"];

    cl_int error = CLFW::get(predicate, "predicate", sizeof(cl_int)* (globalSize));

    error |= kernel->setArg(0, input);
    error |= kernel->setArg(1, predicate);
    error |= kernel->setArg(2, index);
    error |= kernel->setArg(3, compared);

    error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
    return error;
  };

  cl_int UniquePredicate(cl::Buffer &input, cl::Buffer &predicate, cl_int globalSize) {
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["UniquePredicateKernel"];
    
    cl_int error = kernel->setArg(0, input);
           error |= kernel->setArg(1, predicate);
           error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);

    return error;
  }

  cl_int StreamScan_p(cl::Buffer &input, cl::Buffer &result, cl_int globalSize) {
    cl_int error = 0;
    bool isOld;
    cl::Kernel *kernel = &CLFW::Kernels["StreamScanKernel"];
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    int localSize = kernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLFW::DefaultDevice);
    
    cl::Buffer intermediate, intermediateCopy;
    error |= CLFW::get(intermediate, "intermediate", sizeof(cl_int)*(globalSize / localSize));
    error |= CLFW::get(intermediateCopy, "intermediateCopy", sizeof(cl_int)*(globalSize / localSize), isOld);

    if (!isOld) error |= queue->enqueueFillBuffer<cl_int>(intermediateCopy, { -1 }, 0, sizeof(cl_int) * globalSize / localSize);
    
    int currentNumWorkgroups = (globalSize / localSize);
    error |= queue->enqueueCopyBuffer(intermediateCopy, intermediate, 0, 0, sizeof(cl_int)*currentNumWorkgroups);
    error |= kernel->setArg(0, input);
    error |= kernel->setArg(1, result);
    error |= kernel->setArg(2, intermediate);
    error |= kernel->setArg(3, cl::__local(localSize*sizeof(Index)));
    error |= kernel->setArg(4, cl::__local(localSize*sizeof(Index)));
    error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize));
    return error;
  };

  //cl_int StreamScan_s(unsigned int* buffer, unsigned int* result, const int size) {
  //  int nextPowerOfTwo = (int)pow(2, ceil(log(size) / log(2)));
  //  int intermediate = -1;
  //  unsigned int* localBuffer;
  //  unsigned int* scratch;
  //  unsigned int sum = 0;

  //  localBuffer = (unsigned int*)malloc(sizeof(unsigned int)* nextPowerOfTwo);
  //  scratch = (unsigned int*)malloc(sizeof(unsigned int)* nextPowerOfTwo);
  //  //INIT
  //  for (int i = 0; i < size; i++)
  //    StreamScan_Init(buffer, localBuffer, scratch, i, i);
  //  for (int i = size; i < nextPowerOfTwo; ++i)
  //    localBuffer[i] = scratch[i] = 0;

  //  //Add not necessary with only one workgroup.
  //  //Adjacent sync not necessary with only one workgroup.

  //  //SCAN
  //  for (unsigned int i = 1; i < nextPowerOfTwo; i <<= 1) {
  //    for (int j = 0; j < nextPowerOfTwo; ++j) {
  //      HillesSteelScan(localBuffer, scratch, j, i);
  //    }
  //    __local unsigned int *tmp = scratch;
  //    scratch = localBuffer;
  //    localBuffer = tmp;
  //  }
  //  for (int i = 0; i < size; ++i) {
  //    result[i] = localBuffer[i];
  //  }
  //  free(localBuffer);
  //  free(scratch);

  //  return CL_SUCCESS;
  //}

  cl_int SingleCompact(cl::Buffer &input, cl::Buffer &result, cl::Memory &predicate, cl::Buffer &address, cl_int globalSize) {
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["BUSingleCompactKernel"];

    cl_int error  = kernel->setArg(0, input);
           error |= kernel->setArg(1, result);
           error |= kernel->setArg(2, predicate);
           error |= kernel->setArg(3, address);

    error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
    return error;
  }

  cl_int DoubleCompact(cl::Buffer &input, cl::Buffer &result, cl::Buffer &predicate, cl::Buffer &address, cl_int globalSize) {
    cl_int error = 0;
    bool isOld;
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["BUCompactKernel"];
    cl::Buffer zeroBUBuffer;

    error |= CLFW::get(zeroBUBuffer, "zeroBUBuffer", sizeof(BigUnsigned)*globalSize, isOld);
    if (!isOld) {
      BigUnsigned zero;
      initBlkBU(&zero, 0);
      error |= queue->enqueueFillBuffer<BigUnsigned>(zeroBUBuffer, { zero }, 0, globalSize*sizeof(BigUnsigned));
    }
    error |= queue->enqueueCopyBuffer(zeroBUBuffer, result, 0, 0, sizeof(BigUnsigned) * globalSize);

    error |= kernel->setArg(0, input);
    error |= kernel->setArg(1, result);
    error |= kernel->setArg(2, predicate);
    error |= kernel->setArg(3, address);
    error |= kernel->setArg(4, globalSize);
    error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
    return error;
  };

  cl_int UniqueSorted(cl::Buffer &input, cl_int &size) {
    int globalSize = nextPow2(size);
    cl_int error = 0;
    
    cl::Buffer predicate, address, intermediate, result;
    error  = CLFW::get(predicate, "predicate", sizeof(cl_int)*(globalSize));
    error |= CLFW::get(address, "address", sizeof(cl_int)*(globalSize));
    error |= CLFW::get(result, "result", sizeof(BigUnsigned) * globalSize);
    
    error |= UniquePredicate(input, predicate, globalSize);
    error |= StreamScan_p(predicate, address, globalSize);
    error |= SingleCompact(input, result, predicate, address, globalSize);

    input = result;
    
    error |= CLFW::DefaultQueue.enqueueReadBuffer(address, CL_TRUE, (sizeof(cl_int)*globalSize - (sizeof(cl_int))), sizeof(cl_int), &size);
    return error;
  }

  cl_int RadixSortBigUnsigned(cl::Buffer input, cl_int size, cl_int mbits) {
    cl_int error = 0;
    const size_t globalSize = nextPow2(size);

    cl::Buffer predicate, address, bigUnsignedTemp, temp;
    error |= CLFW::get(address, "address", sizeof(cl_int)*(globalSize));
    error |= CLFW::get(bigUnsignedTemp, "bigUnsignedTemp", sizeof(BigUnsigned)*globalSize);

    if (error != CL_SUCCESS) return error;

    //For each bit
    for (unsigned int index = 0; index < mbits; index++) {
      //Predicate the 0's and 1's
      error |= BitPredicate(input, predicate, index, 1, globalSize);

      //Scan the predication buffers.
      error |= StreamScan_p(predicate, address, globalSize);

      //Compacting
      error |= DoubleCompact(input, bigUnsignedTemp, predicate, address, globalSize);

      //Swap result with input.
      temp = input;
      input = bigUnsignedTemp;
      bigUnsignedTemp = temp;
    }
    return error;
  }

  //cl_int BuildBinaryRadixTree_p(cl_int size, cl_int mbits) {
  //  const size_t globalWorkSize[] = { nextPow2(size), 0, 0 };
  //  cl_int error = initBinaryRadixTreeBuffers(globalWorkSize[0]);

  //  error |= clSetKernelArg(Kernels["BuildBinaryRadixTreeKernel"], 0, sizeof(cl_mem), buffers.internalNodes->getBufferPtr());
  //  error |= clSetKernelArg(Kernels["BuildBinaryRadixTreeKernel"], 1, sizeof(cl_mem), buffers.bigUnsignedInput->getBufferPtr());
  //  error |= clSetKernelArg(Kernels["BuildBinaryRadixTreeKernel"], 2, sizeof(cl_int), &mbits);
  //  error |= clSetKernelArg(Kernels["BuildBinaryRadixTreeKernel"], 3, sizeof(cl_int), &size);

  //  if (error == CL_SUCCESS)
  //    error |= clEnqueueNDRangeKernel(CLFW::Queues[0], Kernels["BuildBinaryRadixTreeKernel"], 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
  //  
  //  return error;
  //}

  //cl_int BuildBinaryRadixTree_s(cl_int size, cl_int mbits, BigUnsigned* zpoints, BrtNode* I) {
  //  for (int i = 0; i < size-1; ++i) {
  //    if (i == size - 1) 
  //      cout << size;
  //    BuildBinaryRadixTree(I, zpoints, mbits, size, i);
  //  }
  //  return CL_SUCCESS;
  //}

  //cl_int ComputeLocalSplits_p(cl_int size) {
  //  const size_t globalWorkSize[] = { nextPow2(size), 0, 0 };
  //  cl_int error = clSetKernelArg(Kernels["ComputeLocalSplitsKernel"], 0, sizeof(cl_mem), buffers.localSplits->getBufferPtr());
  //  error |= clSetKernelArg(Kernels["ComputeLocalSplitsKernel"], 1, sizeof(cl_mem), buffers.internalNodes->getBufferPtr());
  //  error |= clSetKernelArg(Kernels["ComputeLocalSplitsKernel"], 2, sizeof(cl_int), &size);
  //  error |= clEnqueueCopyBuffer(CLFW::Queues[0], buffers.localSplitsCopy->getBuffer(), buffers.localSplits->getBuffer(), 
  //    0, 0, sizeof(unsigned int) * globalWorkSize[0], 0, NULL, NULL);
  //  error = clEnqueueNDRangeKernel(CLFW::Queues[0], Kernels["ComputeLocalSplitsKernel"], 1, 0, globalWorkSize, NULL, 0, NULL, NULL);
  //  return error;
  //}

  //cl_int ComputeLocalSplits_s(const cl_int size, vector<unsigned int> local_splits, vector<BrtNode> I) {
  //  if (size > 0) {
  //    local_splits[0] = 1 + I[0].lcp_length / DIM;
  //  }
  //  for (int i = 0; i < size - 1; ++i) {
  //    ComputeLocalSplits(local_splits.data(), I.data(), i);
  //  }
  //  return CL_SUCCESS;
  //}
  //
  //cl_int BinaryRadixToOctree_p(cl_int size, vector<OctNode> octree) {
  //  vector<unsigned int> localSplits(size - 1, 0);
  //  vector<unsigned int> prefixSums(size);
  //  int nextPowerOfTwo = nextPow2(size);
  //  cl_int error = initBinaryRadixToOctreeBuffers(nextPowerOfTwo);

  //  //compute local splits
  //  error |= ComputeLocalSplits_p(size);

  //  //scan the splits
  //  error |= StreamScan_p(buffers.localSplits->getBuffer(), buffers.intermediate->getBuffer(), buffers.intermediateCopy->getBuffer(), buffers.scannedSplits->getBuffer(), nextPowerOfTwo);


  //  //Read in the required octree size
  //  int* temp = (int*)clEnqueueMapBuffer(CLFW::Queues[0], buffers.scannedSplits->getBuffer(), CL_TRUE, CL_MAP_READ, sizeof(int)*(size - 2), sizeof(int), 0, NULL, NULL, NULL); //SLOW!!!
  //  int octreeSize = *temp;//= prefix_sums[n - 1];
  //  error |= clEnqueueUnmapMemObject(CLFW::Queues[0], buffers.scannedSplits->getBuffer(), temp, 0, NULL, NULL);

  //  //Make it a power of two.
  //  int nextOctreeSizePowerOfTwo = nextPow2(octreeSize);

  //  //Create an octree buffer.
  //  if (!isBufferUsable(buffers.octree, sizeof(OctNode)* (nextOctreeSizePowerOfTwo)))
  //    error |= createBuffer(buffers.octree, sizeof(OctNode)* (nextOctreeSizePowerOfTwo));

  //  //use the scanned splits & brt to create octree.
  //  const size_t globalWorkSize[] = { nextPowerOfTwo, 0, 0 };
  //  error |= clSetKernelArg(Kernels["BRT2OctreeKernel_init"], 0, sizeof(cl_mem), buffers.internalNodes->getBufferPtr());
  //  error |= clSetKernelArg(Kernels["BRT2OctreeKernel_init"], 1, sizeof(cl_mem), buffers.octree->getBufferPtr());
  //  error |= clSetKernelArg(Kernels["BRT2OctreeKernel_init"], 2, sizeof(cl_mem), buffers.localSplits->getBufferPtr());
  //  error |= clSetKernelArg(Kernels["BRT2OctreeKernel_init"], 3, sizeof(cl_mem), buffers.scannedSplits->getBufferPtr());
  //  error |= clSetKernelArg(Kernels["BRT2OctreeKernel_init"], 4, sizeof(cl_int), &size);
  //  error |= clEnqueueNDRangeKernel(CLFW::Queues[0], Kernels["BRT2OctreeKernel_init"], 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
  //  error |= clSetKernelArg(Kernels["BRT2OctreeKernel"], 0, sizeof(cl_mem), buffers.internalNodes->getBufferPtr());
  //  error |= clSetKernelArg(Kernels["BRT2OctreeKernel"], 1, sizeof(cl_mem), buffers.octree->getBufferPtr());
  //  error |= clSetKernelArg(Kernels["BRT2OctreeKernel"], 2, sizeof(cl_mem), buffers.localSplits->getBufferPtr());
  //  error |= clSetKernelArg(Kernels["BRT2OctreeKernel"], 3, sizeof(cl_mem), buffers.scannedSplits->getBufferPtr());
  //  error |= clSetKernelArg(Kernels["BRT2OctreeKernel"], 4, sizeof(cl_int), &size);
  //  error |= clEnqueueNDRangeKernel(CLFW::Queues[0], Kernels["BRT2OctreeKernel"], 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
  //  

  //  octree.resize(octreeSize);
  //  OctNode* tempOctree = (OctNode*)clEnqueueMapBuffer(CLFW::Queues[0], buffers.octree->getBuffer(), CL_TRUE, CL_MAP_READ, 0, sizeof(OctNode)*(octreeSize), 0, NULL, NULL, NULL);
  //  memcpy(octree.data(), tempOctree, sizeof(OctNode)*(octreeSize));
  //  error |= clEnqueueUnmapMemObject(CLFW::Queues[0], buffers.octree->getBuffer(), tempOctree, 0, NULL, NULL);

  //  return error;
  //}

  //cl_int BinaryRadixToOctree_s(cl_int size, vector<BrtNode> I, vector<OctNode> octree) {
  //  vector<unsigned int> localSplits(size);
  //  ComputeLocalSplits_s(size, localSplits, I);

  //  vector<unsigned int> prefixSums(size);
  //  StreamScan_s(localSplits.data(), prefixSums.data(), size);

  //  const int octreeSize = prefixSums[prefixSums.size() - 1];
  //  octree.resize(octreeSize);

  //  for (int i = 0; i < octreeSize; ++i)
  //    brt2octree_init(i, octree.data());
  //  for (int brt_i = 1; brt_i < size - 1; ++brt_i)
  //    brt2octree(brt_i, I.data(), octree.data(), localSplits.data(), prefixSums.data(), size, octreeSize);
  //  
  //  return CL_SUCCESS;
  //}
}