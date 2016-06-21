#pragma once
#include "Kernels.h"
namespace Kernels {

  bool benchmarking = false;
  Timer timer;

  void startBenchmark(string benchmarkName) {
    if (benchmarking) {
      timer.restart(benchmarkName);
    }
  }
  void stopBenchmark() {
    if (benchmarking) {
      timer.stop();
      CLFW::DefaultQueue.finish();
    }
  }

  int nextPow2(int num) { return max((int)pow(2, ceil(log(num) / log(2))), 8); }

  inline std::string buToString(BigUnsigned bu) {
    std::string representation = "";
    if (bu.len == 0)
    {
      representation += "[0]";
    }
    else {
      for (int i = bu.len; i > 0; --i) {
        representation += "[" + std::to_string(bu.blk[i - 1]) + "]";
      }
    }

    return representation;
  }

  cl_int UploadPoints(const vector<intn> &points, cl::Buffer &pointsBuffer) {
    startBenchmark("Uploading points");
    cl_int error = 0;
    cl_int roundSize = nextPow2(points.size());
    error |= CLFW::get(pointsBuffer, "pointsBuffer", sizeof(intn)*roundSize);
    error |= CLFW::DefaultQueue.enqueueWriteBuffer(pointsBuffer, CL_TRUE, 0, sizeof(cl_int2) * points.size(), points.data());
    stopBenchmark();
    return error;
  }

  cl_int PointsToMorton_p(cl::Buffer &points, cl::Buffer &zpoints, cl_int size, cl_int bits) {
    startBenchmark("PointsToMorton_p");
    cl_int error = 0;
    size_t globalSize = nextPow2(size);
    error |= CLFW::get(zpoints, "zpoints", globalSize * sizeof(BigUnsigned));
    cl::Kernel kernel = CLFW::Kernels["PointsToMortonKernel"];
    error |= kernel.setArg(0, zpoints);
    error |= kernel.setArg(1, points);
    error |= kernel.setArg(2, size);
    error |= kernel.setArg(3, bits);
    error |= CLFW::DefaultQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(nextPow2(size)), cl::NullRange);
    stopBenchmark();
    return error;
  };
  
  cl_int PointsToMorton_s(cl_int size, cl_int bits, cl_int2* points, BigUnsigned* result) {
    startBenchmark("PointsToMorton_s");
    int nextPowerOfTwo = nextPow2(size);
    for (int gid = 0; gid < nextPowerOfTwo; ++gid) {
      if (gid < size) {
        xyz2z(&result[gid], points[gid], bits);
      }
      else {
        initBlkBU(&result[gid], 0);
      }
    }
    stopBenchmark();
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
    int localSize = std::min((int)kernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLFW::DefaultDevice), globalSize);
    int currentNumWorkgroups = (globalSize / localSize) + 1;
    
    cl::Buffer intermediate, intermediateCopy;
    error |= CLFW::get(intermediate, "intermediate", sizeof(cl_int) * currentNumWorkgroups);
    error |= CLFW::get(intermediateCopy, "intermediateCopy", sizeof(cl_int) * currentNumWorkgroups, isOld);

    if (!isOld) error |= queue->enqueueFillBuffer<cl_int>(intermediateCopy, { -1 }, 0, sizeof(cl_int) * currentNumWorkgroups);
    error |= queue->enqueueCopyBuffer(intermediateCopy, intermediate, 0, 0, sizeof(cl_int) * currentNumWorkgroups);
    error |= kernel->setArg(0, input);
    error |= kernel->setArg(1, result);
    error |= kernel->setArg(2, intermediate);
    error |= kernel->setArg(3, cl::__local(localSize*sizeof(cl_int)));
    error |= kernel->setArg(4, cl::__local(localSize*sizeof(cl_int)));
    error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize));
    return error;
  };

  cl_int StreamScan_s(unsigned int* buffer, unsigned int* result, const int size) {
    int nextPowerOfTwo = (int)pow(2, ceil(log(size) / log(2)));
    int intermediate = -1;
    unsigned int* localBuffer;
    unsigned int* scratch;
    unsigned int sum = 0;

    localBuffer = (unsigned int*)malloc(sizeof(unsigned int)* nextPowerOfTwo);
    scratch = (unsigned int*)malloc(sizeof(unsigned int)* nextPowerOfTwo);
    //INIT
    for (int i = 0; i < size; i++)
      StreamScan_Init(buffer, localBuffer, scratch, i, i);
    for (int i = size; i < nextPowerOfTwo; ++i)
      localBuffer[i] = scratch[i] = 0;

    //Add not necessary with only one workgroup.
    //Adjacent sync not necessary with only one workgroup.

    //SCAN
    for (unsigned int i = 1; i < nextPowerOfTwo; i <<= 1) {
      for (int j = 0; j < nextPowerOfTwo; ++j) {
        HillesSteelScan(localBuffer, scratch, j, i);
      }
      unsigned int *tmp = scratch;
      scratch = localBuffer;
      localBuffer = tmp;
    }
    for (int i = 0; i < size; ++i) {
      result[i] = localBuffer[i];
    }
    free(localBuffer);
    free(scratch);

    return CL_SUCCESS;
  }

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
    startBenchmark("UniqueSorted");
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
    stopBenchmark();
    return error;
  }

  cl_int checkOrder(cl::Buffer &input, cl_bool &inOrder) {
    //Allocate shared memory for each thread block
    //  //Compute elements id
    //  id ← threadIdx.x + threadIdx.y ∗ blockSize
    //  for all threads inside thread block do
    //    //Reading one value per thread to the shared memory
    //    shared[threadIdx.x] ← data[tid];
    //if threadIdx.x = 0 then
    //  Read the next element to the last data inside block
    //  end if
    //  end for
    //  //Wait for all threads to finish reading
    //  syncthreads()
    //  //Perform order checking
    //  shared[id] ←(shared[i] > shared[i + 1]);
    //Perform optimized reduction on shared array
    //  Write out reduction result to global array
  }

  cl_int RadixSortBigUnsigned(cl::Buffer &input, cl_int size, cl_int mbits) {
    startBenchmark("RadixSortBigUnsigned");
    cl_int error = 0;
    const size_t globalSize = nextPow2(size);

    cl::Buffer predicate, address, bigUnsignedTemp, temp;
    error |= CLFW::get(address, "address", sizeof(cl_int)*(globalSize));
    error |= CLFW::get(bigUnsignedTemp, "bigUnsignedTemp", sizeof(BigUnsigned)*globalSize);

    if (error != CL_SUCCESS) return error;

    //For each bit
    for (unsigned int index = 0; index < mbits; index++) {
      //Predicate the 0's and 1's
      error |= BitPredicate(input, predicate, index, 0, globalSize);

      //Scan the predication buffers.
      error |= StreamScan_p(predicate, address, globalSize);

      //Compacting
      error |= DoubleCompact(input, bigUnsignedTemp, predicate, address, globalSize);

      //Swap result with input.
      temp = input;
      input = bigUnsignedTemp;
      bigUnsignedTemp = temp;
    }
    stopBenchmark();
    return error;
  }

  cl_int fourWayScanWithShuffle() {
    //Allocate 4 counting arrays, cnt[4]
    //  Read data in to shared - memory block s_data
    //  for all threadId in thread block do
    //    Extract 2 bits combination from the input data[id]
    //    for b = 0 to 3 do
    //      cnt[b][threadId] ← (b == extract_bits)
    //      end for
    //      Built four way sum tree, Algorithm 5
    //      Clear the last element of each counting array
    //      Down sweep the tree, scan in place Algorithm 6
    //      synchthread
    //      Shuffle data in the shared memory
    //      s_data[cnt[extract_bits][threadId]] ← data
    //      synchthread
    //      Output local sorted data to global memory
    //      end for
  }

  cl_int getBlockSumArray() {

  }

  cl_int FourWayRadixSortBigUnsigned(cl::Buffer &input, cl_int size, cl_int mbits) {
    cl_int error = 0; 
    cl_bool inOrder = false;

    //set blocksize
    cl_int globalSize = nextPow2(size);
    cl_int localSize = min(globalSize, 256);

    //Allocate and initialize 4 block-sum array on GPU
    cl::Buffer blockSum;
    error |= CLFW::get(blockSum, "blockSum", sizeof(cl_int)*(globalSize));

    for (unsigned int bit = 0; bit < mbits; bit += 2) {
      //Perform order checking
      error |= checkOrder(input, inOrder);
      if (inOrder) break;

      //Perform shared-memory 4-way local scan.
      fourWayScanWithShuffle();
      
      //Output total number of each scan path to block sum array
      getBlockSumArray();

      //Perform scan prefix sum on the block sum array
      //scan(blockSum);

      //Compute addresses
      //computeAddresses();

      //Map element to the right position.
     // mapElements();
    }

    return CL_SUCCESS;
  }

  cl_int BuildBinaryRadixTree_p(cl::Buffer &zpoints, cl::Buffer &internalBRTNodes, cl_int size, cl_int mbits) {
    startBenchmark("BuildBinaryRadixTree_p");
    cl::Kernel &kernel = CLFW::Kernels["BuildBinaryRadixTreeKernel"];
    cl::CommandQueue &queue = CLFW::DefaultQueue;
    cl_int globalSize = nextPow2(size);

    cl_int error = CLFW::get(internalBRTNodes, "internalBRTNodes", sizeof(BrtNode)* (globalSize));

    error |= kernel.setArg(0, internalBRTNodes);
    error |= kernel.setArg(1, zpoints);
    error |= kernel.setArg(2, mbits);
    error |= kernel.setArg(3, size);
    error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
    stopBenchmark();
    return error;
  }

  cl_int BuildBinaryRadixTree_s(BigUnsigned* zpoints, BrtNode* internalBRTNodes, cl_int size, cl_int mbits) {
    startBenchmark("BuildBinaryRadixTree_s");
    for (int i = 0; i < size-1; ++i) {
      BuildBinaryRadixTree(internalBRTNodes, zpoints, mbits, size, i);
    }
    stopBenchmark();
    return CL_SUCCESS;
  }

  cl_int ComputeLocalSplits_p(cl::Buffer &internalBRTNodes, cl::Buffer &localSplits, cl_int size) {
    startBenchmark("ComputeLocalSplits_p");
    cl_int globalSize = nextPow2(size);
    cl::Kernel &kernel = CLFW::Kernels["ComputeLocalSplitsKernel"];
    cl::CommandQueue &queue = CLFW::DefaultQueue;

    bool isOld;
    cl::Buffer zeroBuffer;

    cl_int error  = CLFW::get(localSplits, "localSplits", sizeof(cl_int) * globalSize);
           error |= CLFW::get(zeroBuffer, "zeroBuffer", sizeof(cl_int) * globalSize, isOld);

    //Fill any new zero buffers with zero. Then initialize localSplits with zero.
    if (!isOld) {
      cl_int zero = 0;
      error |= queue.enqueueFillBuffer<cl_int>(zeroBuffer, { zero }, 0, sizeof(cl_int) * globalSize);
    }
    error |= queue.enqueueCopyBuffer(zeroBuffer, localSplits, 0, 0, sizeof(cl_int) * globalSize);

    error |= kernel.setArg(0, localSplits);
    error |= kernel.setArg(1, internalBRTNodes);
    error |= kernel.setArg(2, size);

    error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
    stopBenchmark();
    return error;
  }

  cl_int ComputeLocalSplits_s(vector<BrtNode> &I, vector<cl_uint> &local_splits, const cl_int size) {
    startBenchmark("ComputeLocalSplits_s");
    if (size > 0) {
      local_splits[0] = 1 + I[0].lcp_length / DIM;
    }
    for (int i = 0; i < size - 1; ++i) {
      ComputeLocalSplits(local_splits.data(), I.data(), i);
    }
    stopBenchmark();
    return CL_SUCCESS;
  }

  cl_int InitOctree(cl::Buffer &internalBRTNodes, cl::Buffer &octree, cl::Buffer &localSplits, cl::Buffer &scannedSplits, cl_int size, cl_int octreeSize) {
    startBenchmark("InitOctree");
    cl_int globalSize = nextPow2(octreeSize);
    cl::Kernel &kernel = CLFW::Kernels["BRT2OctreeKernel_init"];
    cl::CommandQueue &queue = CLFW::DefaultQueue;
    cl_int error = 0;

    error |= kernel.setArg(0, internalBRTNodes);
    error |= kernel.setArg(1, octree);
    error |= kernel.setArg(2, localSplits);
    error |= kernel.setArg(3, scannedSplits);
    error |= kernel.setArg(4, size);

    error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
    stopBenchmark();
    return error;
  }

  cl_int BinaryRadixToOctree_p(cl::Buffer &internalBRTNodes, vector<OctNode> &octree_vec, cl_int size) {
    startBenchmark("BinaryRadixToOctree_p");
    int globalSize = nextPow2(size);
    cl::Kernel &kernel = CLFW::Kernels["BRT2OctreeKernel"];
    cl::CommandQueue &queue = CLFW::DefaultQueue;

    cl::Buffer localSplits, scannedSplits, octree;
    cl_int error = CLFW::get(scannedSplits, "scannedSplits", sizeof(cl_int) * globalSize);

    error |= ComputeLocalSplits_p(internalBRTNodes, localSplits, size);
    error |= StreamScan_p(localSplits, scannedSplits, globalSize);

    //Read in the required octree size
    cl_int octreeSize;
    error |= CLFW::DefaultQueue.enqueueReadBuffer(scannedSplits, CL_TRUE, sizeof(int)*(size - 2), sizeof(int), &octreeSize);
    cl_int roundOctreeSize = nextPow2(octreeSize);

    //Create an octree buffer.
    error |= CLFW::get(octree, "octree", sizeof(OctNode) * roundOctreeSize);

    //use the scanned splits & brt to create octree.
    InitOctree(internalBRTNodes, octree, localSplits, scannedSplits, size, octreeSize);

    error |= kernel.setArg(0, internalBRTNodes);
    error |= kernel.setArg(1, octree);
    error |= kernel.setArg(2, localSplits);
    error |= kernel.setArg(3, scannedSplits);
    error |= kernel.setArg(4, size);

    error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);

    octree_vec.resize(octreeSize);
    error |= queue.enqueueReadBuffer(octree, CL_TRUE, 0, sizeof(OctNode)*octreeSize, octree_vec.data());
    stopBenchmark();
    return error;
  }

  cl_int BinaryRadixToOctree_s(vector<BrtNode> &internalBRTNodes, vector<OctNode> &octree, cl_int size) {
    startBenchmark("BinaryRadixToOctree_s");
    vector<unsigned int> localSplits(size);
    ComputeLocalSplits_s(internalBRTNodes, localSplits, size);

    vector<unsigned int> prefixSums(size);
    StreamScan_s(localSplits.data(), prefixSums.data(), size);

    const int octreeSize = prefixSums[size - 1];
    octree.resize(octreeSize);
    for (int i = 0; i < octreeSize; ++i)
      brt2octree_init(i, octree.data());
    for (int brt_i = 1; brt_i < size - 1; ++brt_i)
      brt2octree(brt_i, internalBRTNodes.data(), octree.data(), localSplits.data(), prefixSums.data(), size, octreeSize);
    stopBenchmark();
    return CL_SUCCESS;
  }

  cl_int BuildOctree_s(const vector<intn>& points, vector<OctNode> &octree, int bits, int mbits) {
    if (points.empty()) {
      throw logic_error("Zero points not supported");
      return -1;
    }
    int numPoints = points.size();
    int roundNumPoints = Kernels::nextPow2(points.size());
    vector<BigUnsigned> zpoints(roundNumPoints);

    //Points to Z Order
    Kernels::PointsToMorton_s(points.size(), bits, (cl_int2*)points.data(), zpoints.data());

    //Sort and unique Z points
    sort(zpoints.rbegin(), zpoints.rend(), weakCompareBU);
    numPoints = unique(zpoints.begin(), zpoints.end(), weakEqualsBU) - zpoints.begin();

    //Build BRT
    vector<BrtNode> I(numPoints - 1);
    Kernels::BuildBinaryRadixTree_s(zpoints.data(), I.data(), numPoints, mbits);

    //Build Octree
    Kernels::BinaryRadixToOctree_s(I, octree, numPoints);
    return CL_SUCCESS;
  }

  cl_int BuildOctree_p(const vector<intn>& points, vector<OctNode> &octree, int bits, int mbits) {
    system("cls");
    if (points.empty())
      throw logic_error("Zero points not supported");

    int size = points.size();
    cl_int error = 0;
    cl::Buffer pointsBuffer, zpoints, internalBRTNodes;
    error |= Kernels::UploadPoints(points, pointsBuffer);
    error |= Kernels::PointsToMorton_p(pointsBuffer, zpoints, size, bits);
    error |= Kernels::RadixSortBigUnsigned(zpoints, size, mbits);
    error |= Kernels::UniqueSorted(zpoints, size);
    error |= Kernels::BuildBinaryRadixTree_p(zpoints, internalBRTNodes, size, mbits);
    error |= Kernels::BinaryRadixToOctree_p(internalBRTNodes, octree, size);
    return error;
  }
}