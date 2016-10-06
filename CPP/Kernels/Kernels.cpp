#pragma once
#include "Kernels.h"

#ifdef __OPENCL_VERSION__
#define VEC_ACCESS(v, a) (v.a)
#define X_(v) VEC_ACCESS(v, x)
#define Y_(v) VEC_ACCESS(v, y)
#define Z_(v) VEC_ACCESS(v, z)
#else
#define VEC_ACCESS(v, a) (v.s[a])
#define X_(v) VEC_ACCESS(v, 0)
#define Y_(v) VEC_ACCESS(v, 1)
#define Z_(v) VEC_ACCESS(v, 2)
#endif

/* Testing methods */
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
}

/* Uploaders */
namespace Kernels {
	cl_int UploadKarrasPoints(const vector<float_2> &points, cl::Buffer &karrasPointsBuffer) {
		startBenchmark("Uploading Karras points");
		cl_int error = 0;
		cl_int roundSize = nextPow2(points.size());
		error |= CLFW::get(karrasPointsBuffer, "karrasPointsBuffer", sizeof(float_2)*roundSize);
		error |= CLFW::DefaultQueue.enqueueWriteBuffer(karrasPointsBuffer, CL_TRUE, 0, sizeof(float_2) * points.size(), points.data());
		stopBenchmark();
		return error;
	}

  cl_int UploadQuantizedPoints(const vector<int_n> &points, cl::Buffer &pointsBuffer) {
    startBenchmark("Uploading points");
    cl_int error = 0;
    cl_int roundSize = nextPow2(points.size());
    error |= CLFW::get(pointsBuffer, "pointsBuffer", sizeof(int_n)*roundSize);
    error |= CLFW::DefaultQueue.enqueueWriteBuffer(pointsBuffer, CL_TRUE, 0, sizeof(int_n) * points.size(), points.data());
    stopBenchmark();
    return error;
  }

  cl_int UploadLines(const vector<Line> &lines, cl::Buffer &linesBuffer) {
    startBenchmark("Uploading lines");
    cl_int error = 0;
    cl_int roundSize = nextPow2(lines.size());
    error |= CLFW::get(linesBuffer, "linesBuffer", sizeof(Line)*roundSize);
    error |= CLFW::DefaultQueue.finish();
    error |= CLFW::DefaultQueue.enqueueWriteBuffer(linesBuffer, CL_TRUE, 0, sizeof(Line) * lines.size(), lines.data());
    stopBenchmark();
    return error;
  }
}

/* Downloaders */ 
namespace Kernels {
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

  cl_int DownloadPoints(vector<int_n> &points, cl::Buffer &pointsBuffer, cl_int size) {
    startBenchmark("Downloading points");
    cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(pointsBuffer, CL_TRUE, 0, sizeof(int_2) * size, points.data());
    stopBenchmark();
    return error;
  }

  cl_int DownloadZPoints(vector<BigUnsigned> &zpoints, cl::Buffer &zpointsBuffer, cl_int size) {
    startBenchmark("Downloading points");
    cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(zpointsBuffer, CL_TRUE, 0, sizeof(BigUnsigned) * size, zpoints.data());
    stopBenchmark();
    return error;
  }

	cl_int DownloadConflictPairs(vector<ConflictPair> &conflictPairsVec, cl::Buffer &conflictPairsBuffer, cl_int size) {
		startBenchmark("Downloading conflict pairs");
		conflictPairsVec.resize(size);
		cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(conflictPairsBuffer, CL_TRUE, 0, sizeof(ConflictPair) * size, conflictPairsVec.data());
		stopBenchmark();
		return error;
	}
}

/* Reduce Kernels */
namespace Kernels {
  cl_int AddAll(cl::Buffer &numbers, cl_uint& gpuSum, cl_int size) {
    //startBenchmark("AddAll kernel");
    cl_int nextPowerOfTwo = nextPow2(size);
    if (nextPowerOfTwo != size) return CL_INVALID_ARG_SIZE;
    cl::Kernel &kernel = CLFW::Kernels["reduce"];
    cl::CommandQueue &queue = CLFW::DefaultQueue;

    //Each thread processes 2 items.
    int globalSize = size / 2;

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
}

/* Predication Kernels */
namespace Kernels {
  cl_int BitPredicate(cl::Buffer &input, cl::Buffer &predicate, unsigned int &index, unsigned char compared, cl_int globalSize) {
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["BitPredicateKernel"];

    cl_int error = CLFW::get(predicate, "bitPredicate", sizeof(cl_int)* (globalSize));

    error |= kernel->setArg(0, input);
    error |= kernel->setArg(1, predicate);
    error |= kernel->setArg(2, index);
    error |= kernel->setArg(3, compared);
    error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
    return error;
  };

  cl_int LinePredicate(cl::Buffer &input, cl::Buffer &predicate, unsigned int &index, unsigned char compared, cl_int size, cl_int mbits) {
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["LinePredicateKernel"];
    int roundSize = nextPow2(size);
    cl_int error = CLFW::get(predicate, "linePredicate", sizeof(cl_int)* (roundSize));

    error |= kernel->setArg(0, input);
    error |= kernel->setArg(1, predicate);
    error |= kernel->setArg(2, index);
    error |= kernel->setArg(3, compared);
    error |= kernel->setArg(4, mbits);
    error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
    return error;
  };

  cl_int LevelPredicate(cl::Buffer &input, cl::Buffer &predicate, unsigned int &index, unsigned char compared, cl_int size, cl_int mbits) {
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["LevelPredicateKernel"];
    int roundSize = nextPow2(size);
    cl_int error = CLFW::get(predicate, "levelPredicate", sizeof(cl_int)* (roundSize));

    error |= kernel->setArg(0, input);
    error |= kernel->setArg(1, predicate);
    error |= kernel->setArg(2, index);
    error |= kernel->setArg(3, compared);
    error |= kernel->setArg(4, mbits);
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
    error |= kernel->setArg(2, cl::__local(localSize * sizeof(BigUnsigned)));
    error |= kernel->setArg(3, cl::__local(localSize * 4 * sizeof(cl_int)));
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
  
  cl_int UniquePredicate(cl::Buffer &input, cl::Buffer &predicate, cl_int globalSize) {
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["UniquePredicateKernel"];

    cl_int error = kernel->setArg(0, input);
    error |= kernel->setArg(1, predicate);
    error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);

    return error;
  }
}

/* Compaction Kernels */
namespace Kernels {
  cl_int SingleCompact(cl::Buffer &input, cl::Buffer &result, cl::Memory &predicate, cl::Buffer &address, cl_int globalSize) {
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    cl::Kernel *kernel = &CLFW::Kernels["BUSingleCompactKernel"];

    cl_int error = kernel->setArg(0, input);
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
      error |= queue->enqueueFillBuffer<BigUnsigned>(zeroBUBuffer, { zero }, 0, globalSize * sizeof(BigUnsigned));
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
      error |= queue->enqueueFillBuffer<Line>(zeroLineBuffer, { zero }, 0, roundSize * sizeof(Line));
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
}

/* Scan/Prefix Sum Kernels */
namespace Kernels {
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

  cl_int StreamScan_p(cl::Buffer &input, cl::Buffer &result, cl_int globalSize, string intermediateName) {
    cl_int error = 0;
    bool isOld;
    cl::Kernel *kernel = &CLFW::Kernels["StreamScanKernel"];
    cl::CommandQueue *queue = &CLFW::DefaultQueue;
    int localSize = std::min((int)kernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLFW::DefaultDevice), globalSize);
    int currentNumWorkgroups = (globalSize / localSize) + 1;

    cl::Buffer intermediate, intermediateCopy;
    error |= CLFW::get(intermediate, intermediateName, sizeof(cl_int) * currentNumWorkgroups);
    error |= CLFW::get(intermediateCopy, intermediateName + "copy", sizeof(cl_int) * currentNumWorkgroups, isOld);

    if (!isOld) error |= queue->enqueueFillBuffer<cl_int>(intermediateCopy, { -1 }, 0, sizeof(cl_int) * currentNumWorkgroups);
    error |= queue->enqueueCopyBuffer(intermediateCopy, intermediate, 0, 0, sizeof(cl_int) * currentNumWorkgroups);
    error |= kernel->setArg(0, input);
    error |= kernel->setArg(1, result);
    error |= kernel->setArg(2, intermediate);
    error |= kernel->setArg(3, cl::__local(localSize * sizeof(cl_int)));
    error |= kernel->setArg(4, cl::__local(localSize * sizeof(cl_int)));
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
}

/* Sorts */
namespace Kernels {
  cl_int RadixSortBigUnsigned_p(cl::Buffer &input, cl::Buffer &result, cl_int size, cl_int mbits) {
    cl_int error = 0;
    const size_t globalSize = nextPow2(size);

    cl::Buffer predicate, address, bigUnsignedTemp, temp;
    error |= CLFW::get(address, "BUAddress", sizeof(cl_int)*(globalSize));
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
      if (error != CL_SUCCESS)
        return error;
      //Scan the predication buffers.
      error |= StreamScan_p(predicate, address, globalSize, "radixSortBUIntermediate");
      if (error != CL_SUCCESS)
        return error;
      //Compacting
      error |= DoubleCompact(result, bigUnsignedTemp, predicate, address, globalSize);
      if (error != CL_SUCCESS)
        return error;
      //Swap result with input.
      temp = result;
      result = bigUnsignedTemp;
      bigUnsignedTemp = temp;
    }
    stopBenchmark();
    return error;
  }

  cl_int RadixSortLines_p(cl::Buffer &input, cl::Buffer &sortedLines, cl_int size, cl_int mbits) {
    cl_int error = 0;

    vector<Line> lines;
    lines.resize(size);
    error |= CLFW::DefaultQueue.enqueueReadBuffer(input, CL_TRUE, 0, size * sizeof(Line), lines.data());

    const size_t globalSize = nextPow2(size);

    cl::Buffer predicate, address, tempLinesBuffer, temp;
    error |= CLFW::get(address, "lineAddress", sizeof(cl_int)*(globalSize));
    error |= CLFW::get(tempLinesBuffer, "tempLinesBuffer", sizeof(Line)*globalSize);
    error |= CLFW::get(sortedLines, "sortedLines", sizeof(Line)*globalSize);
    error |= CLFW::DefaultQueue.enqueueCopyBuffer(input, sortedLines, 0, 0, sizeof(Line) * globalSize);

    if (error != CL_SUCCESS)
      return error;

    //For each bit
    startBenchmark("RadixSortBigUnsigned");
    for (unsigned int index = 0; index < mbits; index++) {
      //Predicate the 0's and 1's
      error |= LinePredicate(sortedLines, predicate, index, 0, size, mbits);

      //Scan the predication buffers.
      error |= StreamScan_p(predicate, address, globalSize, "radixSortLineIntermediate");

      //Compacting
      error |= LineDoubleCompact(sortedLines, tempLinesBuffer, predicate, address, size);

      //Swap result with input.
      temp = sortedLines;
      sortedLines = tempLinesBuffer;
      tempLinesBuffer = temp;
      error |= CLFW::DefaultQueue.enqueueReadBuffer(sortedLines, CL_TRUE, 0, size * sizeof(Line), lines.data());
    }

    for (unsigned int index = 0; index < mbits / DIM; index++) {
      //Predicate the 0's and 1's
      error |= LevelPredicate(sortedLines, predicate, index, 0, size, mbits);

      //Scan the predication buffers.
      error |= StreamScan_p(predicate, address, globalSize, "radixSortLineIntermediate");

      //Compacting
      error |= LineDoubleCompact(sortedLines, tempLinesBuffer, predicate, address, size);

      //Swap result with input.
      temp = sortedLines;
      sortedLines = tempLinesBuffer;
      tempLinesBuffer = temp;
    }

    stopBenchmark();
    return error;
  }
}

/* Tree Building Kernels */
namespace Kernels {
  cl_int ComputeLocalSplits_p(cl::Buffer &internalBRTNodes, cl::Buffer &localSplits, cl_int size) {
    startBenchmark("ComputeLocalSplits_p");
    cl_int globalSize = nextPow2(size);
    cl::Kernel &kernel = CLFW::Kernels["ComputeLocalSplitsKernel"];
    cl::CommandQueue &queue = CLFW::DefaultQueue;

    bool isOld;
    cl::Buffer zeroBuffer;

    cl_int error = CLFW::get(localSplits, "localSplits", sizeof(cl_int) * globalSize);
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
    for (int i = 0; i < size - 1; ++i) {
      BuildBinaryRadixTree(internalBRTNodes, zpoints, mbits, size, i);
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
    error |= StreamScan_p(localSplits, scannedSplits, globalSize, "BinaryRadixToOctreeIntermediate");

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

  cl_int BuildOctree_s(const vector<int_n>& points, vector<OctNode> &octree, int bits, int mbits) {
    if (points.empty()) {
      throw logic_error("Zero points not supported");
      return -1;
    }
    int numPoints = points.size();
    int roundNumPoints = nextPow2(points.size());
    vector<BigUnsigned> zpoints(roundNumPoints);

    //Points to Z Order
    PointsToMorton_s(points.size(), bits, (int_n*)points.data(), zpoints.data());

    //Sort and unique Z points
    sort(zpoints.rbegin(), zpoints.rend(), weakCompareBU);
    numPoints = unique(zpoints.begin(), zpoints.end(), weakEqualsBU) - zpoints.begin();

    //Build BRT
    vector<BrtNode> I(numPoints - 1);
    BuildBinaryRadixTree_s(zpoints.data(), I.data(), numPoints, mbits);

    //Build Octree
    BinaryRadixToOctree_s(I, octree, numPoints);
    return CL_SUCCESS;
  }

  cl_int BuildOctree_p(cl::Buffer zpoints, cl_int numZPoints, vector<OctNode> &octree, int bits, int mbits) {
    if (benchmarking)
      system("cls");

    int currentSize = numZPoints;
    cl_int error = 0;
    cl::Buffer sortedZPoints, internalBRTNodes;
    error |= RadixSortBigUnsigned_p(zpoints, sortedZPoints, currentSize, mbits);
    error |= UniqueSorted(sortedZPoints, currentSize);
    error |= BuildBinaryRadixTree_p(sortedZPoints, internalBRTNodes, currentSize, mbits);
    error |= BinaryRadixToOctree_p(internalBRTNodes, octree, currentSize);
    return error;
  }
}

/* Ambiguous cell resolution kernels */
namespace Kernels {
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
      OctNode node = octree[boundingBoxes[i]];
      lines[i].level = (short)node.level;

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

  unsigned char getQuadrant(BigUnsigned *lcp, unsigned char lcpShift, unsigned char i) {
    BigUnsigned buMask, result;
    unsigned int quadrantMask = (DIM == 2) ? 3 : 7;
    initBlkBU(&buMask, quadrantMask);

    shiftBULeft(&buMask, &buMask, i * DIM + lcpShift);
    andBU(&result, &buMask, lcp);
    shiftBURight(&result, &result, i * DIM + lcpShift);
    return (result.len == 0) ? 0 : result.blk[0];
  }

  float_n getNodeCenterFromLCP(BigUnsigned *LCP, unsigned int LCPLength, float octreeWidth) {
    unsigned int level = LCPLength / DIM;
    unsigned int lcpShift = LCPLength % DIM;
    float_n center = { 0.0, 0.0 };
    float centerShift = octreeWidth / (2.0 * (1 << level));

    for (int i = 0; i < level; ++i) {
      unsigned quadrant = getQuadrant(LCP, lcpShift, i);
      X_(center) += (quadrant & (1 << 0)) ? centerShift : -centerShift;
      Y_(center) += (quadrant & (1 << 1)) ? centerShift : -centerShift;
#if DIM == 3
      center.z += (quadrant & (1 << 2)) ? centerShift : -centerShift;
#endif
      centerShift *= 2.0;
    }
    return center;
  }

  cl_int SortLinesByLvlThenVal_p(vector<Line> &unorderedLines, cl::Buffer &sortedLinesBuffer, cl::Buffer &zpoints, const Resln &resln) {
    //Two lines are required for an ambigous cell to appear.
    if (unorderedLines.size() < 2) return CL_INVALID_ARG_SIZE;
    cl_int error = 0;
    cl::Buffer linesBuffer;
    error |= UploadLines(unorderedLines, linesBuffer);
    if (error != CL_SUCCESS)
      UploadLines(unorderedLines, linesBuffer);
    error |= ComputeLineLCPs_p(linesBuffer, zpoints, unorderedLines.size(), resln.mbits);
    error |= RadixSortLines_p(linesBuffer, sortedLinesBuffer, unorderedLines.size(), resln.mbits);
    return error;
  }

  cl_int FindConflictCells_s(cl::Buffer sortedLinesBuffer, cl_int numLines, cl::Buffer octreeBuffer, OctNode* octree,
    unsigned int numOctNodes, float_n octreeCenter, float octreeWidth, vector<ConflictPair> &conflictPairs, float_2* points) {
    //Two lines are required for an ambigous cell to appear.
    if (numLines < 2) return CL_INVALID_ARG_SIZE;
    cl_int error = 0;
    vector<int> boundingBoxes;
    cl::Buffer boundingBoxesBuffer;
    error |= ComputeLineBoundingBoxes_p(sortedLinesBuffer, octreeBuffer, boundingBoxesBuffer, numLines);
    error |= DownloadBoundingBoxes(boundingBoxesBuffer, boundingBoxes, numLines);

    vector<Line> sortedLines(numLines);
    error |= DownloadLines(sortedLinesBuffer, sortedLines, numLines);
    ConflictPair initialPair;
    initialPair.i[0] = initialPair.i[1] = -1;
    conflictPairs.resize(4 * numOctNodes, initialPair);
    for (unsigned int i = 0; i < numOctNodes; ++i) {
      FindConflictCells(octree, numOctNodes, octreeCenter, octreeWidth, conflictPairs.data(),
        boundingBoxes.data(), boundingBoxes.size(), sortedLines.data(), sortedLines.size(), points, i);
    }
    return error;
  }
	
	cl_int FindConflictCells_p(cl::Buffer &sortedLinesBuffer, cl_int numLines, cl::Buffer &octreeBuffer,
		unsigned int numOctNodes, float_n &octreeCenter, float octreeWidth, cl::Buffer &conflictPairs, cl::Buffer &points) {
		//Two lines are required for an ambigous cell to appear.
		if (numLines < 2) return CL_INVALID_ARG_SIZE;
		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl::Kernel &kernel = CLFW::Kernels["FindConflictCellsKernel"];
		
		cl::Buffer boundingBoxesBuffer, initialConflictPairsBuffer;
		cl_int error = ComputeLineBoundingBoxes_p(sortedLinesBuffer, octreeBuffer, boundingBoxesBuffer, numLines); //needs a better name

		bool isOld;
		ConflictPair initialPair;
		initialPair.i[0] = initialPair.i[1] = -1;
		
		error |= CLFW::get(conflictPairs, "conflictPairs", 4 * nextPow2(numOctNodes) * sizeof(ConflictPair));
		error |= CLFW::get(initialConflictPairsBuffer, "initialConflictPairs", 4 * nextPow2(numOctNodes) * sizeof(ConflictPair), isOld);
		if (!isOld) {
			error |= queue.enqueueFillBuffer<ConflictPair>(initialConflictPairsBuffer, { initialPair }, 0, 4 * nextPow2(numOctNodes) * sizeof(ConflictPair));
		}
		error |= queue.enqueueCopyBuffer(initialConflictPairsBuffer, conflictPairs, 0, 0, 4 * nextPow2(numOctNodes) * sizeof(ConflictPair));

		error |= kernel.setArg(0, octreeBuffer);
		error |= kernel.setArg(1, points);
		error |= kernel.setArg(2, sortedLinesBuffer);
		error |= kernel.setArg(3, boundingBoxesBuffer); //rename to SCC
		error |= kernel.setArg(4, conflictPairs);
		error |= kernel.setArg(5, numOctNodes);
		error |= kernel.setArg(6, numLines); //Pretty sure numSCCS = numLines
		error |= kernel.setArg(7, numLines);
		error |= kernel.setArg(8, octreeCenter);
		error |= kernel.setArg(9, octreeWidth);
		error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numOctNodes), cl::NullRange);
		
		return error;
	}
}

/* Hybrid Kernels */
namespace Kernels {
  cl_int UniqueSorted(cl::Buffer &input, cl_int &size) {
    startBenchmark("UniqueSorted");
    int globalSize = nextPow2(size);
    cl_int error = 0;

    cl::Buffer predicate, address, intermediate, result;
    error = CLFW::get(predicate, "predicate", sizeof(cl_int)*(globalSize));
    error |= CLFW::get(address, "address", sizeof(cl_int)*(globalSize));
    error |= CLFW::get(result, "result", sizeof(BigUnsigned) * globalSize);

    error |= UniquePredicate(input, predicate, globalSize);
    error |= StreamScan_p(predicate, address, globalSize, "UniqueIntermediate");
    error |= SingleCompact(input, result, predicate, address, globalSize);

    input = result;

    error |= CLFW::DefaultQueue.enqueueReadBuffer(address, CL_TRUE, (sizeof(cl_int)*globalSize - (sizeof(cl_int))), sizeof(cl_int), &size);
    stopBenchmark();
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
}

/* Morton Kernels */ 
namespace Kernels {
  cl_int PointsToMorton_p(cl::Buffer &points, cl::Buffer &zpoints, cl_int size, cl_int bits) {
    cl_int error = 0;
    size_t globalSize = nextPow2(size);
    bool old;
    error |= CLFW::get(zpoints, "zpoints", globalSize * sizeof(BigUnsigned), old, CLFW::DefaultContext, CL_MEM_READ_ONLY);
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

  cl_int PointsToMorton_s(cl_int size, cl_int bits, int_n* points, BigUnsigned* result) {
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
}
