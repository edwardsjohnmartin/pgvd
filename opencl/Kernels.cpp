﻿#pragma once
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
      CLFW::DefaultQueue.finish();
      timer.stop();
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

  cl_int UploadLines(const vector<Line> &lines, cl::Buffer &linesBuffer) {
    startBenchmark("Uploading lines");
    cl_int error = 0;
    cl_int roundSize = nextPow2(lines.size());
    error |= CLFW::get(linesBuffer, "linesBuffer", sizeof(Line)*roundSize);
    error |= CLFW::DefaultQueue.enqueueWriteBuffer(linesBuffer, CL_TRUE, 0, sizeof(Line) * lines.size(), lines.data());
    stopBenchmark();
    return error;
  }

  cl_int DownloadLines(cl::Buffer &linesBuffer, vector<Line> &lines, cl_int size) {
    startBenchmark("Downloading lines");
    lines.resize(size);
    cl_int error = 0;
    error |= CLFW::DefaultQueue.enqueueReadBuffer(linesBuffer, CL_TRUE, 0, sizeof(Line)*size, lines.data());
    stopBenchmark();
    return error;
  }

  cl_int DownloadBoundingBoxes(cl::Buffer &boundingBoxesBuffer, vector<int> &boundingBoxes, cl_int size) {
    startBenchmark("Downloading boundingBoxes");
    boundingBoxes.resize(size);
    cl_int error = 0;
    error |= CLFW::DefaultQueue.enqueueReadBuffer(boundingBoxesBuffer, CL_TRUE, 0, sizeof(int)*size, boundingBoxes.data());
    stopBenchmark();
    return error;
  }

  cl_int PointsToMorton_p(cl::Buffer &points, cl::Buffer &zpoints, cl_int size, cl_int bits) {
    cl_int error = 0;
    size_t globalSize = nextPow2(size);
    error |= CLFW::get(zpoints, "zpoints", globalSize * sizeof(BigUnsigned));
    cl::Kernel kernel = CLFW::Kernels["PointsToMortonKernel"];
    error |= kernel.setArg(0, zpoints);
    error |= kernel.setArg(1, points);
    error |= kernel.setArg(2, size);
    error |= kernel.setArg(3, bits);
    startBenchmark("PointsToMorton_p");
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
  
  cl_int LinePredicate(cl::Buffer &input, cl::Buffer &predicate, unsigned int &index, unsigned char compared, cl_int size) {
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["LinePredicateKernel"];
    int roundSize = nextPow2(size);
    cl_int error = CLFW::get(predicate, "predicate", sizeof(cl_int)* (roundSize));

    error |= kernel->setArg(0, input);
    error |= kernel->setArg(1, predicate);
    error |= kernel->setArg(2, index);
    error |= kernel->setArg(3, compared);
    error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
    return error;
  };

  cl_int GetTwoBitMask_p(cl::Buffer &input, cl::Buffer &masks, unsigned int index, unsigned char compared, cl_int size) {
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["GetTwoBitMaskKernel"];
    int globalSize = nextPow2(size);
    int localSize = std::min((int)kernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLFW::DefaultDevice), globalSize);

    cl_int error = CLFW::get(masks, "masks", sizeof(cl_int) * (globalSize) * 4);
    
    error |= kernel->setArg(0, input);
    error |= kernel->setArg(1, masks);
    error |= kernel->setArg(2, cl::__local(localSize*sizeof(BigUnsigned)));
    error |= kernel->setArg(3, cl::__local(localSize *4*sizeof(cl_int)));
    error |= kernel->setArg(4, index);
    error |= kernel->setArg(5, compared);
    error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
    return error;
  }
  
  cl_int GetTwoBitMask_s(BigUnsigned* input, unsigned int *masks, unsigned int index, unsigned char compared, cl_int size) {
    for (int i = 0; i < size; ++i) {
      GetTwoBitMask(input, masks, index, compared, i);
    }
    return CL_SUCCESS;
  }

  cl_int GetFourWayPrefixSum_p(cl::Buffer &input, cl::Buffer &masks, unsigned int index, unsigned char compared, cl_int size)
  {
    return -1;
  }

  cl_int GetFourWayPrefixSum_s(BigUnsigned* input, unsigned int *fourWayPrefix, unsigned int index, unsigned char compared, cl_int size)
  {
    vector<unsigned int> masks(size * 4);

    for (int i = 0; i < size; ++i) {
      GetTwoBitMask(input, masks.data(), index, compared, i);
    }

    fourWayPrefix[0] = fourWayPrefix[1] = fourWayPrefix[2] = fourWayPrefix[3] = 0;
    for (int i = 1; i < size; ++i) 
    {
      fourWayPrefix[i * 4] = masks[(i - 1) * 4] + fourWayPrefix[(i - 1) * 4];
      fourWayPrefix[i * 4 + 1] = masks[(i - 1) * 4 + 1] + fourWayPrefix[(i - 1) * 4 + 1];
      fourWayPrefix[i * 4 + 2] = masks[(i - 1) * 4 + 2] + fourWayPrefix[(i - 1) * 4 + 2];
      fourWayPrefix[i * 4 + 3] = masks[(i - 1) * 4 + 3] + fourWayPrefix[(i - 1) * 4 + 3];
    }
    
    return CL_SUCCESS;
  }

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

  cl_int LineDoubleCompact(cl::Buffer &input, cl::Buffer &result, cl::Buffer &predicate, cl::Buffer &address, cl_int globalSize) {
    cl_int error = 0;
    bool isOld;
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["LineCompactKernel"];
    cl::Buffer zeroLineBuffer;
    unsigned int roundSize = nextPow2(globalSize);
    error |= CLFW::get(zeroLineBuffer, "zeroLineBuffer", sizeof(Line)*roundSize, isOld);
    if (!isOld) {
      Line zero;
      initBlkBU(&zero.lcp, 0);
      zero.lcpLength = -1;
      error |= queue->enqueueFillBuffer<Line>(zeroLineBuffer, { zero }, 0, roundSize*sizeof(Line));
    }
    error |= queue->enqueueCopyBuffer(zeroLineBuffer, result, 0, 0, sizeof(Line) * globalSize);

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

  cl_int CheckOrder_s(cl::Buffer &input, cl_int &size) {
    int nextPowerOfTwo = (int)pow(2, ceil(log(size) / log(2)));
    cl_bool *predicate;

    //Allocate shared memory for each thread block.
    predicate = (cl_bool*)malloc(sizeof(cl_bool) * size);

    //compute elements id
    for (cl_int id = 0; id < nextPowerOfTwo; ++id) {

    }

    //sync threads
    for (cl_int id = 0; id < nextPowerOfTwo; ++id) {

    }

    //Perform optimized reduction on shared array (algorithm by Harris)
    //Write out reduction result to global array

    free(predicate);
    return CL_SUCCESS;
  }

  cl_int RadixSortBigUnsigned(cl::Buffer &input, cl::Buffer &result, cl_int size, cl_int mbits) {
    cl_int error = 0;
    const size_t globalSize = nextPow2(size);

    cl::Buffer predicate, address, bigUnsignedTemp, temp;
    error |= CLFW::get(address, "address", sizeof(cl_int)*(globalSize));
    error |= CLFW::get(bigUnsignedTemp, "bigUnsignedTemp", sizeof(BigUnsigned)*globalSize);
    error |= CLFW::get(result, "sortedZPoints", sizeof(BigUnsigned)*globalSize);
    error |= CLFW::DefaultQueue.enqueueCopyBuffer(input, result, 0, 0, sizeof(BigUnsigned) * globalSize);

    if (error != CL_SUCCESS) 
      return error;
    cl_uint test;
    cl_uint sum;
    //For each bit
    startBenchmark("RadixSortBigUnsigned");
    for (unsigned int index = 0; index < mbits; index++) {
      //Predicate the 0's and 1's
      error |= BitPredicate(result, predicate, index, 0, globalSize);

      //Scan the predication buffers.
      error |= StreamScan_p(predicate, address, globalSize);
      
      //Compacting
      error |= DoubleCompact(result, bigUnsignedTemp, predicate, address, globalSize);
      
      //Swap result with input.
      temp = result;
      result = bigUnsignedTemp;
      bigUnsignedTemp = temp;
    }
    stopBenchmark();
    return error;
  }

  cl_int RadixSortLines(cl::Buffer &input, cl::Buffer &sortedLines, cl_int size, cl_int mbits) {
    cl_int error = 0;

    /*vector<Line> lines;
    lines.resize(size);
    error |= CLFW::DefaultQueue.enqueueReadBuffer(input, CL_TRUE, 0, size*sizeof(Line), lines.data());
    cout << "Before:" << endl;
    for (int i = 0; i < size; ++i) {
      cout << buToString(lines[i].lcp, lines[i].lcpLength) << endl;
    }*/


    const size_t globalSize = nextPow2(size);

    cl::Buffer predicate, address, tempLinesBuffer, temp;
    error |= CLFW::get(address, "address", sizeof(cl_int)*(globalSize));
    error |= CLFW::get(tempLinesBuffer, "tempLinesBuffer", sizeof(Line)*globalSize);
    error |= CLFW::get(sortedLines, "sortedLines", sizeof(Line)*globalSize);
    error |= CLFW::DefaultQueue.enqueueCopyBuffer(input, sortedLines, 0, 0, sizeof(Line) * globalSize);

    if (error != CL_SUCCESS)
      return error;

    //For each bit
    startBenchmark("RadixSortBigUnsigned");
    for (unsigned int index = 0; index < mbits; index++) {
      //Predicate the 0's and 1's
      error |= LinePredicate(sortedLines, predicate, index, 0, size);

      //Scan the predication buffers.
      error |= StreamScan_p(predicate, address, globalSize);

      //Compacting
      error |= LineDoubleCompact(sortedLines, tempLinesBuffer, predicate, address, size);

      //Swap result with input.
      temp = sortedLines;
      sortedLines = tempLinesBuffer;
      tempLinesBuffer = temp;
    }

    /*error |= CLFW::DefaultQueue.enqueueReadBuffer(sortedLines, CL_TRUE, 0, size*sizeof(Line), lines.data());
    cout << "After:" << endl;
    for (int i = 0; i < size; ++i) {
      cout << buToString(lines[i].lcp, lines[i].lcpLength)<<endl;
    }*/

    stopBenchmark();
    return error;
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
    octree[0].parent = -1;
    octree[0].level = 0;
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
    if (benchmarking)
      system("cls");
    if (points.empty())
      throw logic_error("Zero points not supported");

    int size = points.size();
    cl_int error = 0;
    cl::Buffer pointsBuffer, zpoints, sortedZPoints, internalBRTNodes;
    error |= Kernels::UploadPoints(points, pointsBuffer);
    error |= Kernels::PointsToMorton_p(pointsBuffer, zpoints, size, bits);
    error |= Kernels::RadixSortBigUnsigned(zpoints, sortedZPoints, size, mbits);
    error |= Kernels::UniqueSorted(sortedZPoints, size);
    error |= Kernels::BuildBinaryRadixTree_p(sortedZPoints, internalBRTNodes, size, mbits);
    error |= Kernels::BinaryRadixToOctree_p(internalBRTNodes, octree, size);
    return error;
  }

  cl_int AddAll(cl::Buffer &numbers, cl_uint& gpuSum, cl_int size) {
    //startBenchmark("AddAll kernel");
    cl_int nextPowerOfTwo = nextPow2(size);
    if (nextPowerOfTwo != size) return CL_INVALID_ARG_SIZE;
    cl::Kernel &kernel = CLFW::Kernels["reduce"];
    cl::CommandQueue &queue = CLFW::DefaultQueue;

    //Each thread processes 2 items.
    int globalSize = size/2;

    //Brent's theorem
    //int itemsPerThread = nextPow2(log(globalSize));
    //globalSize /= itemsPerThread;

    int suggestedLocal = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLFW::DefaultDevice);
    int localSize = std::min(globalSize, suggestedLocal);

    cl::Buffer reduceResult;
    cl_int resultSize = nextPow2(size / localSize);
    cl_int error = CLFW::get(reduceResult, "reduceResult", resultSize * sizeof(cl_uint));

    error |= kernel.setArg(0, numbers);
    error |= kernel.setArg(1, cl::__local(localSize * sizeof(cl_uint)));
    error |= kernel.setArg(2, nextPowerOfTwo);
    error |= kernel.setArg(3, reduceResult);
    error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize));

    //If multiple workgroups ran, we need to do a second level reduction.
    if (suggestedLocal <= globalSize) {
      error |= kernel.setArg(0, reduceResult);
      error |= kernel.setArg(1, cl::__local(localSize * sizeof(cl_uint)));
      error |= kernel.setArg(2, resultSize);
      error |= kernel.setArg(3, reduceResult);
      error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(localSize / 2), cl::NDRange(localSize / 2));
    }

    error |= queue.enqueueReadBuffer(reduceResult, CL_TRUE, 0, sizeof(cl_uint), &gpuSum);
    //stopBenchmark();
    return error;
  }

  cl_int CheckOrder(cl::Buffer &numbers, cl_uint& gpuSum, cl_int size) {
    if (size < 0) return CL_INVALID_ARG_SIZE;
    else if (size < 2) {
      gpuSum = 0;
      return CL_SUCCESS;
    }
    //startBenchmark("CheckOrder kernel");
    cl_int nextPowerOfTwo = pow(2, ceil(log(size) / log(2)));
    cl::Kernel &kernel = CLFW::Kernels["CheckOrder"];
    cl::CommandQueue &queue = CLFW::DefaultQueue;

    //Each thread processes 2 items in the reduce.
    int globalSize = nextPowerOfTwo / 2;

    //Brent's theorem
   /* int itemsPerThread = pow(2, ceil(log(log(globalSize)) / log(2)));
    if ((itemsPerThread < globalSize) && itemsPerThread != 0) {
      globalSize /= itemsPerThread;
    }*/

    int suggestedLocal = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLFW::DefaultDevice);
    int localSize = std::min(globalSize, suggestedLocal);

    cl::Buffer reduceResult;
    cl_int resultSize = nextPow2(nextPowerOfTwo / localSize);
    cl_int error = CLFW::get(reduceResult, "reduceResult", resultSize * sizeof(cl_uint));

    error |= kernel.setArg(0, numbers);
    error |= kernel.setArg(1, cl::__local(localSize * sizeof(cl_uint)));
    error |= kernel.setArg(2, size);
    error |= kernel.setArg(3, nextPowerOfTwo);
    error |= kernel.setArg(4, reduceResult);
    error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize));

    //If multiple workgroups ran, we need to do a second level reduction.
    if (suggestedLocal <= globalSize) {
      cl::Kernel &kernel = CLFW::Kernels["reduce"];
      error |= kernel.setArg(0, reduceResult);
      error |= kernel.setArg(1, cl::__local(localSize * sizeof(cl_uint)));
      error |= kernel.setArg(2, resultSize);
      error |= kernel.setArg(3, reduceResult);
      error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(localSize / 2), cl::NDRange(localSize / 2));
    }
    error |= queue.enqueueReadBuffer(reduceResult, CL_TRUE, 0, sizeof(cl_uint), &gpuSum);
   // stopBenchmark();
    return error;
  }

  cl_int ComputeLineLCPs_s(Line* lines, BigUnsigned* zpoints, cl_int size, int mbits) {
    for (int i = 0; i < size; ++i) {
      calculateLineLCP(lines, zpoints, mbits, i);
    }
    return CL_SUCCESS;
  }

  cl_int ComputeLineLCPs_p(cl::Buffer &linesBuffer, cl::Buffer &zpoints, cl_int size, int mbits) {
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["ComputeLineLCPKernel"];
    cl_int error = 0;
    error |= kernel->setArg(0, linesBuffer);
    error |= kernel->setArg(1, zpoints);
    error |= kernel->setArg(2, mbits);
    error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
    error = CLFW::DefaultQueue.finish();
    return error;
  }

  cl_int ComputeLineBoundingBoxes_s(Line* lines, int* boundingBoxes, OctNode *octree, cl_int numLines) {
    for (int i = 0; i < numLines; ++i) {
      boundingBoxes[i] = getOctNode(lines[i].lcp, lines[i].lcpLength, octree);
    }
    return CL_SUCCESS;
  }

  cl_int ComputeLineBoundingBoxes_p(cl::Buffer &linesBuffer, cl::Buffer &octree, cl::Buffer &boundingBoxes, cl_int numLines) {
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["ComputeLineBoundingBoxesKernel"];
    int roundNumber = nextPow2(numLines);
    cl_int error = CLFW::get(boundingBoxes, "boundingBoxes", sizeof(cl_int)* (roundNumber));

    error |= kernel->setArg(0, linesBuffer);
    error |= kernel->setArg(1, octree);
    error |= kernel->setArg(2, boundingBoxes);
    error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(numLines), cl::NullRange);
    error = CLFW::DefaultQueue.finish();
    return error;
  }

  int compareThingy(unsigned int a_indx, unsigned int b_indx, unsigned int a_lvl, unsigned int b_lvl) {
    if (a_lvl < b_lvl)
      return -1;
    if (b_lvl < a_lvl)
      return 1;
    if (a_indx < b_indx)
      return 1;
    if (b_indx < a_indx)
      return -1;
    return 0;
  }

  int findBoundingBox(unsigned int key, OctNode node, Line* lines, int* boundingBoxes, unsigned int size) {
    unsigned int start = 0;
    unsigned int end = size;
    while (start != end) {
      unsigned int index = ((end - start) / 2) + start;
      int x = compareThingy(key, boundingBoxes[index], node.level, lines[index].lcpLength/DIM);
      if (0 > x) end = index;
      else if (x > 0) start = index + 1;
      else return index;
    }
    return -1;
  }

  cl_int FindAmbiguousCells_s(vector<Line> orderedLines, vector<int> boundingBoxes, vector<OctNode> octree, const unsigned int octreeSize) {
    struct NodeColors { int colors[4] = { -1, -1, -1, -1 }; };
    vector<NodeColors> nodeColors_vec(octree.size());
    //For each internal node (in parallel)
    for (int i = 0; i < octreeSize; ++i) {
      OctNode node = octree[i];
      if (node.leaf != 0) {
        int index = i;
        while (index != -1) {
          node = octree[index];
          int bbIndex = findBoundingBox(index, node, orderedLines.data(), boundingBoxes.data(), orderedLines.size());
          while (bbIndex > 0 && boundingBoxes[bbIndex-1] == bbIndex) bbIndex--;
          if (bbIndex != -1) {
            for (int j = 0; j < 1 << DIM; ++j) {
              if (!(node.leaf & (1 << j))) continue;
              if (true) {
                struct NodeColors &nodeColors = nodeColors_vec[i];
                if (nodeColors.colors[j] == -1)
                  nodeColors.colors[j] = orderedLines[bbIndex].color;
                else if (nodeColors.colors[j] != orderedLines[bbIndex].color) {
                  nodeColors.colors[j] = -2;
                  cout << index <<":"<< j << " is ambiguous" << endl;
                }
              }
            }
          }
          index = node.parent;
        }  
      }
    }
    return CL_SUCCESS;
  }
}