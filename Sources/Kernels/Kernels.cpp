#pragma once
#include "../GLUtilities/gl_utils.h"
#include  "../Kernels/Kernels.h"
#include  "../Options/options.h"

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

/* Testing methods */
namespace Kernels {
    bool benchmarking = false;
log4cplus::Logger benchmarkLogger =
    log4cplus::Logger::getInstance("Kernels.benchmark");
Timer* benchmarkTimer = 0;

    void startBenchmark(string benchmarkName) {
        if (benchmarking) {
          benchmarkTimer = new Timer(benchmarkLogger, benchmarkName);
            // timer.restart(benchmarkName);
        }
    }
    void stopBenchmark() {
        if (benchmarking) {
            CLFW::DefaultQueue.finish();
            // timer.stop();
            benchmarkTimer->stop();
            delete benchmarkTimer;
            benchmarkTimer = 0;
        }
    }

    int nextPow2(int num) { return max((int)pow(2, ceil(log(num) / log(2))), 8); }

    std::string buToString(BigUnsigned bu) {
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

    std::string buToString(BigUnsigned bu, int len) {
        std::string representation = "";
        if (len == 0)
        {
            representation += "NULL";
        }
        else {
            //int shift = len%DIM;
            // len -= shift;
            for (int i = len - 1; i >= 0; --i) {
                representation += std::to_string(getBUBit(&bu, i));
            }
        }
        return representation;
    }
}

std::ostream& operator<<(std::ostream& out, const BrtNode& node) {
    out << node.left << " " << node.left_leaf << " " << node.right_leaf << " " <<
        Kernels::buToString(node.lcp) << " " << node.lcp_length << " " << node.parent;
    return out;
}

/* Uploaders */
namespace Kernels {
    cl_int UploadKarrasPoints(const vector<floatn> &points, cl::Buffer &karrasPointsBuffer) {
        startBenchmark("Uploading Karras points");
        cl_int error = 0;
        cl_int roundSize = nextPow2(points.size());
        error |= CLFW::get(karrasPointsBuffer, "karrasPointsBuffer", sizeof(floatn)*roundSize);
        error |= CLFW::DefaultQueue.enqueueWriteBuffer(karrasPointsBuffer, CL_TRUE, 0, sizeof(floatn) * points.size(), points.data());
        stopBenchmark();
        return error;
    }

    cl_int UploadQuantizedPoints(const vector<intn> &points, cl::Buffer &pointsBuffer) {
        startBenchmark("Uploading points");
        cl_int error = 0;
        cl_int roundSize = nextPow2(points.size());
        error |= CLFW::get(pointsBuffer, "qPoints", sizeof(intn)*roundSize);
        error |= CLFW::DefaultQueue.enqueueWriteBuffer(pointsBuffer, CL_TRUE, 0, sizeof(intn) * points.size(), points.data());
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
}

/* Downloaders */
namespace Kernels {
    //TODO: Fix the order of these parameters...
    cl_int DownloadInts(cl::Buffer &integersBuffer, vector<int> &integers, cl_int size) {
        startBenchmark("Downloading points");
        integers.resize(size);
        cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(integersBuffer, CL_TRUE, 0, sizeof(cl_int) * size, integers.data());
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

    cl_int DownloadFloatnPoints(vector<floatn> &points, cl::Buffer &pointsBuffer, cl_int size) {
        startBenchmark("Downloading points");
        points.resize(size);
        cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(pointsBuffer, CL_TRUE, 0, sizeof(floatn) * size, points.data());
        stopBenchmark();
        return error;
    }

    cl_int DownloadQPoints(vector<intn> &points, cl::Buffer &pointsBuffer, cl_int size) {
        startBenchmark("Downloading points");
        points.resize(size);
        cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(pointsBuffer, CL_TRUE, 0, sizeof(intn) * size, points.data());
        stopBenchmark();
        return error;
    }

    cl_int DownloadZPoints(vector<BigUnsigned> &zpoints, cl::Buffer &zpointsBuffer, cl_int size) {
        startBenchmark("Downloading points");
        zpoints.resize(size);
        cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(zpointsBuffer, CL_TRUE, 0, sizeof(BigUnsigned) * size, zpoints.data());
        stopBenchmark();
        return error;
    }

    cl_int DownloadConflicts(vector<Conflict> &conflictPairsVec, cl::Buffer &conflictPairsBuffer, cl_int size) {
        startBenchmark("Downloading conflict pairs");
        conflictPairsVec.resize(size);
        cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(conflictPairsBuffer, CL_TRUE, 0, sizeof(Conflict) * size, conflictPairsVec.data());
        stopBenchmark();
        return error;
    }
    
    cl_int DownloadFacetPairs(vector<FacetPair> &facetPairsVec, cl::Buffer &facetPairsBuffer, cl_int size) {
        startBenchmark("Downloading facet pairs");
        facetPairsVec.resize(size);
        cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(facetPairsBuffer, CL_TRUE, 0, sizeof(FacetPair) * size, facetPairsVec.data());
        stopBenchmark();
        return error;
    }

    cl_int DownloadBCells(vector<BCell> &BCellsVec, cl::Buffer &BCellsBuffer, cl_int size) {
        startBenchmark("Downloading bCells");
        BCellsVec.resize(size);
        cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(BCellsBuffer, CL_TRUE, 0, sizeof(BCell) * size, BCellsVec.data());
        stopBenchmark();
        return error;
    }

    cl_int DownloadConflictInfo(vector<ConflictInfo> &out, cl::Buffer &in, cl_int size) {
        startBenchmark("Downloading conflict info");
        out.resize(size);
        cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(in, CL_TRUE, 0, sizeof(ConflictInfo) * size, out.data());
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
    }

    cl_int BUBitPredicate(cl::Buffer &input, cl::Buffer &predicate, unsigned int &index, unsigned char compared, cl_int globalSize) {
        cl::CommandQueue *queue = &CLFW::DefaultQueue;
        cl::Kernel *kernel = &CLFW::Kernels["BUBitPredicateKernel"];

        cl_int error = CLFW::get(predicate, "buBitPredicate", sizeof(cl_int)* (globalSize));

        error |= kernel->setArg(0, input);
        error |= kernel->setArg(1, predicate);
        error |= kernel->setArg(2, index);
        error |= kernel->setArg(3, compared);
        error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
        return error;
    };

    cl_int BCellPredicate(
        cl::Buffer &input_i,
        cl::Buffer &predicate_o,
        unsigned int &index_i,
        unsigned char compared,
        cl_int size,
        cl_int mbits)
    {
        cl::CommandQueue *queue = &CLFW::DefaultQueue;
        cl::Kernel *kernel = &CLFW::Kernels["BCellPredicateKernel"];
        int roundSize = nextPow2(size);
        cl_int error = CLFW::get(predicate_o, "bCellPredicate", sizeof(cl_int)* (roundSize));

        error |= kernel->setArg(0, input_i);
        error |= kernel->setArg(1, predicate_o);
        error |= kernel->setArg(2, index_i);
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
    cl_int DoubleCompact(cl::Buffer &input, cl::Buffer &result, cl::Buffer &predicate, cl::Buffer &address, cl_int globalSize) {
        cl_int error = 0;
        bool isOld;
        cl::CommandQueue *queue = &CLFW::DefaultQueue;
        cl::Kernel *kernel = &CLFW::Kernels["CompactKernel"];
        cl::Buffer zeroBUBuffer;

        error |= CLFW::get(zeroBUBuffer, "zeroBuffer", sizeof(cl_int)*globalSize, isOld);
        if (!isOld) {
            error |= queue->enqueueFillBuffer<cl_int>(zeroBUBuffer, { 0 }, 0, globalSize * sizeof(cl_int));
        }
        error |= queue->enqueueCopyBuffer(zeroBUBuffer, result, 0, 0, sizeof(cl_int) * globalSize);

        error |= kernel->setArg(0, input);
        error |= kernel->setArg(1, result);
        error |= kernel->setArg(2, predicate);
        error |= kernel->setArg(3, address);
        error |= kernel->setArg(4, globalSize);
        error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
        return error;
    };

    cl_int BUSingleCompact(cl::Buffer &input, cl::Buffer &result, cl::Memory &predicate, cl::Buffer &address, cl_int globalSize) {
        cl::CommandQueue *queue = &CLFW::DefaultQueue;
        cl::Kernel *kernel = &CLFW::Kernels["BUSingleCompactKernel"];

        cl_int error = kernel->setArg(0, input);
        error |= kernel->setArg(1, result);
        error |= kernel->setArg(2, predicate);
        error |= kernel->setArg(3, address);

        error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
        return error;
    }

    cl_int BUDoubleCompact(cl::Buffer &input, cl::Buffer &result, cl::Buffer &predicate, cl::Buffer &address, cl_int globalSize) {
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

    cl_int BCellFacetDoubleCompact(
        cl::Buffer &inputBCells_i,
        cl::Buffer &inputIndices_i,
        cl::Buffer &resultBCells_i,
        cl::Buffer &resultIndices_i,
        cl::Buffer &predicate_i,
        cl::Buffer &address_i,
        cl_int globalSize)
    {
        cl_int error = 0;
        bool isOld;
        cl::CommandQueue *queue = &CLFW::DefaultQueue;
        cl::Kernel *kernel = &CLFW::Kernels["BCellFacetCompactKernel"];
        cl::Buffer zeroBCellBuffer;
        cl::Buffer zeroIndexBuffer;
        unsigned int roundSize = nextPow2(globalSize);
        error |= CLFW::get(zeroBCellBuffer, "zeroBCellBuffer", sizeof(BCell)*roundSize, isOld);
        error |= CLFW::get(zeroIndexBuffer, "zeroIndexBuffer", sizeof(cl_int)*roundSize, isOld);
        
        
        if (!isOld) {
            BCell zero;
            initBlkBU(&zero.lcp, 0);
            zero.lcpLength = -1;
            zero.padding[0] = zero.padding[1] = zero.padding[2] = 0;
            error |= queue->enqueueFillBuffer<BCell>(zeroBCellBuffer, { zero }, 0, roundSize * sizeof(BCell));
            error |= queue->enqueueFillBuffer<cl_int>(zeroIndexBuffer, { 0 }, 0, roundSize * sizeof(cl_int));
        }
        error |= queue->enqueueCopyBuffer(zeroBCellBuffer, resultBCells_i, 0, 0, sizeof(BCell) * globalSize);
        error |= queue->enqueueCopyBuffer(zeroIndexBuffer, resultIndices_i, 0, 0, sizeof(cl_int) * globalSize);


        error |= kernel->setArg(0, inputBCells_i);
        error |= kernel->setArg(1, resultBCells_i);
        error |= kernel->setArg(2, inputIndices_i);
        error |= kernel->setArg(3, resultIndices_i);
        error |= kernel->setArg(4, predicate_i);
        error |= kernel->setArg(5, address_i);
        error |= kernel->setArg(6, globalSize);
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

cl_int StreamScan_p(
    cl::Buffer &input, cl::Buffer &result, const cl_int globalSize,
    string intermediateName, bool exclusive) {
  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("Kernels.StreamScan_p");
  cl_int error = 0;
  bool isOld;
  cl::Kernel *kernel = &CLFW::Kernels["StreamScanKernel"];
  cl::CommandQueue *queue = &CLFW::DefaultQueue;

  const int wgSize = (int)kernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(
      CLFW::DefaultDevice);
  const int localSize = std::min(wgSize, globalSize);
  int currentNumWorkgroups = (globalSize / localSize) + 1;

  LOG4CPLUS_DEBUG(logger, "wgSize = " << wgSize);
  LOG4CPLUS_DEBUG(logger, "localSize = " << localSize);
  LOG4CPLUS_DEBUG(logger, "globalSize = " << globalSize);

  cl::Buffer intermediate, intermediateCopy;
  error |= CLFW::get(intermediate, intermediateName,
                     sizeof(cl_int) * currentNumWorkgroups);
  error |= CLFW::get(intermediateCopy, intermediateName + "copy",
                     sizeof(cl_int) * currentNumWorkgroups, isOld);

  if (!isOld) {
    error |= queue->enqueueFillBuffer<cl_int>(
        intermediateCopy, { -1 }, 0, sizeof(cl_int) * currentNumWorkgroups);
    assert_cl_error(error);
  }
  error |= queue->enqueueCopyBuffer(
      intermediateCopy, intermediate, 0, 0,
      sizeof(cl_int) * currentNumWorkgroups);
  assert_cl_error(error);
  error |= kernel->setArg(0, input);
  assert_cl_error(error);
  error |= kernel->setArg(1, result);
  assert_cl_error(error);
  error |= kernel->setArg(2, intermediate);
  assert_cl_error(error);
  error |= kernel->setArg(3, cl::__local(localSize * sizeof(cl_int)));
  assert_cl_error(error);
  error |= kernel->setArg(4, cl::__local(localSize * sizeof(cl_int)));
  assert_cl_error(error);
  //error |= kernel->setArg(5, exclusive);
  error |= queue->enqueueNDRangeKernel(
      *kernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize));
  assert_cl_error(error);
  return error;
}

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
//Approx 16% of total build
cl_int RadixSortBigUnsigned_p(
    cl::Buffer &input, cl::Buffer &result, cl_int size, cl_int mbits) {
  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("Kernels.RadixSortBigUnsigned_p");

  cl_int error = 0;
  const size_t globalSize = nextPow2(size);
  cl::Buffer predicate, address, bigUnsignedTemp, temp;
  error |= CLFW::get(address, "BUAddress", sizeof(cl_int)*(globalSize));
  assert_cl_error(error);
  error |= CLFW::get(
      bigUnsignedTemp, "bigUnsignedTemp", sizeof(BigUnsigned)*globalSize);
  assert_cl_error(error);
  error |= CLFW::get(result, "sortedZPoints", sizeof(BigUnsigned)*globalSize);
  assert_cl_error(error);
  error |= CLFW::DefaultQueue.enqueueCopyBuffer(
      input, result, 0, 0, sizeof(BigUnsigned) * globalSize);
  assert_cl_error(error);

  if (error != CL_SUCCESS)
    return error;
  cl_uint test;
  cl_uint sum;
  // cl_int testSize = size;
  cl_int testSize = globalSize;
  //For each bit
  startBenchmark("RadixSortBigUnsigned");
  for (unsigned int index = 0; index < mbits; index++) {
    //Predicate the 0's and 1's
    error |= BUBitPredicate(result, predicate, index, 0, testSize);
    assert(error == CL_SUCCESS);
    // if (error != CL_SUCCESS)
    //   return error;
    //Scan the predication buffers.
    error |= StreamScan_p(
        predicate, address, testSize, "radixSortBUIntermediate");
    assert(error == CL_SUCCESS);
    // if (error != CL_SUCCESS)
    //   return error;
    //Compacting
    error |= BUDoubleCompact(
        result, bigUnsignedTemp, predicate, address, testSize);
    assert(error == CL_SUCCESS);
    // if (error != CL_SUCCESS)
    //   return error;
    //Swap result with input.
    temp = result;
    result = bigUnsignedTemp;
    bigUnsignedTemp = temp;
  }
  stopBenchmark();

  if (logger.isEnabledFor(log4cplus::TRACE_LOG_LEVEL)) {
    vector<BigUnsigned> buf;
    Kernels::DownloadZPoints(buf, result, testSize);
    LOG4CPLUS_TRACE(logger, "OrderedZPoints (" << testSize << ")");
    for (int i = 0; i < testSize; i++) {
      LOG4CPLUS_TRACE(logger, "  " << Kernels::buToString(buf[i]));
    }
  }

  return error;
}

cl_int RadixSortPairsByKey(
    cl::Buffer &unsortedKeys_i,
    cl::Buffer &unsortedValues_i,
    cl::Buffer &sortedKeys_o,
    cl::Buffer &sortedValues_o,
    cl_int size)
{
  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("Kernels.RadixSortPairsByKey");

  cl_int error = 0;
  const size_t globalSize = nextPow2(size);


  cl::Buffer predicate, address, tempKeys, tempValues, swap;
  error |= CLFW::get(address, "radixAddress", sizeof(cl_int)*(globalSize));
  error |= CLFW::get(tempKeys, "tempRadixKeys", sizeof(cl_int)*globalSize);
  error |= CLFW::get(tempValues, "tempRadixValues", sizeof(cl_int)*globalSize);
  error |= CLFW::get(sortedKeys_o, "sortedRadixKeys", sizeof(cl_int)*globalSize);
  error |= CLFW::get(sortedValues_o, "sortedRadixValues", sizeof(cl_int)*globalSize);
  //error |= CLFW::DefaultQueue.enqueueCopyBuffer(unsortedKeys_i, sortedKeys_o, 0, 0, sizeof(cl_int) * size);
  //error |= CLFW::DefaultQueue.enqueueCopyBuffer(unsortedValues_i, sortedValues_o, 0, 0, sizeof(cl_int) * size);
        
  if (error != CL_SUCCESS)
    return error;

  //For each bit
  startBenchmark("RadixSortBigUnsigned");
  for (unsigned int index = 0; index < sizeof(cl_int)*8; index++) {
    //Predicate the 0's and 1's
    error |= BitPredicate(unsortedKeys_i, predicate, index, 0, size);

    //Scan the predication buffers.
    error |= StreamScan_p(predicate, address, globalSize, "RSKBVI");

    //Compacting
    error |= DoubleCompact(unsortedKeys_i, tempKeys, predicate, address, size);
    error |= DoubleCompact(unsortedValues_i, tempValues, predicate, address, size);

    //Swap result with input.
    swap = tempKeys;
    tempKeys = unsortedKeys_i;
    unsortedKeys_i = swap;

    swap = tempValues;
    tempValues = unsortedValues_i;
    unsortedValues_i = swap;
    cl_uint gpuSum;
    // TODO: replace break out code. This was causing differences in running
    // on John's mac between CPU and GPU.
    // if (index % 4 == 0) {
    //     //Checking order is expensive, but can save lots if we can break early.
    //     CheckOrder(unsortedKeys_i, gpuSum, size);
    //     if (gpuSum == 0) break;
    // }
  }



  sortedKeys_o = unsortedKeys_i;
  sortedValues_o = unsortedValues_i;
  stopBenchmark();

  if (logger.isEnabledFor(log4cplus::TRACE_LOG_LEVEL)) {
    vector<cl_int> keys, values;
    DownloadInts(sortedKeys_o, keys, size);
    DownloadInts(sortedValues_o, values, size);
    LOG4CPLUS_TRACE(logger, " Octnode2Facet (" << size << ")");
    for (int i = 0; i < size; i++) {
      LOG4CPLUS_TRACE(logger, keys[i] << " / " << values[i]);
    }
  }

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

    //Approx 9% of build time
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

    //Approx 7% of build time
    cl_int BinaryRadixToOctree_p(cl::Buffer &internalBRTNodes, int &newSize, cl_int size) {
        if (size <= 1) return CL_INVALID_ARG_VALUE;
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
        error |= CLFW::DefaultQueue.enqueueReadBuffer(scannedSplits, CL_TRUE,
            sizeof(cl_int)*(size - 2), sizeof(cl_int), &octreeSize);
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
        newSize = octreeSize;
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
        int roundNumPoints = nextPow2(points.size());
        vector<BigUnsigned> zpoints(roundNumPoints);

        //Points to Z Order
        PointsToMorton_s(points.size(), bits, (intn*)points.data(), zpoints.data());

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

cl_int BuildOctree_p(cl::Buffer zpoints, cl_int numZPoints, int &newSize, int bits, int mbits) {
  int currentSize = numZPoints;
  cl_int error = 0;
  cl::Buffer sortedZPoints, internalBRTNodes;

  error |= RadixSortBigUnsigned_p(zpoints, sortedZPoints, currentSize, mbits);
  assert(error == CL_SUCCESS);
  error |= UniqueSorted(sortedZPoints, currentSize);
  assert(error == CL_SUCCESS);
  error |= BuildBinaryRadixTree_p(sortedZPoints, internalBRTNodes, currentSize, mbits);
  assert(error == CL_SUCCESS);
  error |= BinaryRadixToOctree_p(internalBRTNodes, newSize, currentSize);
  assert(error == CL_SUCCESS);
  return error;
}
}

/* Ambiguous cell resolution kernels */
namespace Kernels {
    cl_int GetBCellLCP_s(
        Line* lines,
        BigUnsigned* zpoints,
        BCell* bCells,
        cl_int* facetIndices,
        cl_int size,
        int mbits)
    {
        for (int i = 0; i < size; ++i) {
            GetBCellLCP(lines, zpoints, bCells, facetIndices, mbits, i);
        }
        return CL_SUCCESS;
    }

    cl_int GetBCellLCP_p(
        cl::Buffer &linesBuffer_i,
        cl::Buffer &zpoints_i,
        cl::Buffer &bCells_o,
        cl::Buffer &facetIndices_o,
        cl_int size,
        int mbits)
    {
        cl::CommandQueue *queue = &CLFW::DefaultQueue;
        cl::Kernel *kernel = &CLFW::Kernels["GetBCellLCPKernel"];
        cl_int error = 0;

        cl_int roundSize = nextPow2(size);
        error |= CLFW::get(bCells_o, "BCells", roundSize * sizeof(BCell));
        error |= CLFW::get(facetIndices_o, "facetIndices", roundSize * sizeof(cl_int));
        error |= kernel->setArg(0, linesBuffer_i);
        error |= kernel->setArg(1, zpoints_i);
        error |= kernel->setArg(2, bCells_o);
        error |= kernel->setArg(3, facetIndices_o);
        error |= kernel->setArg(4, mbits);
        error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

        return error;
    }

    cl_int LookUpOctnodeFromBCell_s(BCell* bCells, OctNode *octree, int* BCellToOctree, cl_int numBCells) {
        for (int i = 0; i < numBCells; ++i) {
            BCellToOctree[i] = getOctNode(bCells[i].lcp, bCells[i].lcpLength, octree);
        }
        return CL_SUCCESS;
    }

cl_int LookUpOctnodeFromBCell_p(
    cl::Buffer &bCells_i, cl::Buffer &octree_i, cl::Buffer &BCellToOctnode,
    cl_int numBCells) {

  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("Kernels.LookUpOctnodeFromBCell_p");

  cl::CommandQueue *queue = &CLFW::DefaultQueue;
  cl::Kernel *kernel = &CLFW::Kernels["LookUpOctnodeFromBCellKernel"];
  int roundNumber = nextPow2(numBCells);
  cl_int error = CLFW::get(
      BCellToOctnode, "BCellToOctnode", sizeof(cl_int)* (roundNumber));

  error |= kernel->setArg(0, bCells_i);
  error |= kernel->setArg(1, octree_i);
  error |= kernel->setArg(2, BCellToOctnode);
  error |= queue->enqueueNDRangeKernel(
      *kernel, cl::NullRange, cl::NDRange(numBCells), cl::NullRange);
  error = CLFW::DefaultQueue.finish();

  if (logger.isEnabledFor(log4cplus::TRACE_LOG_LEVEL)) {
    vector<cl_int> octNodeIndices;
    DownloadInts(BCellToOctnode, octNodeIndices, numBCells);
    LOG4CPLUS_TRACE(logger, "BCellToOctnode (" << numBCells << ")");
    for (int i = 0; i < numBCells; i++) {
      LOG4CPLUS_TRACE(logger, octNodeIndices[i]);
    }
  }

  return error;
}

    cl_int GetFacetPairs_s(cl_int* BCellToOctree, FacetPair *facetPairs, cl_int numLines) {
        for (int i = 0; i < numLines; ++i) {
            int leftNeighbor = (i == 0) ? -1 : BCellToOctree[i - 1];
            int rightNeighbor = (i == numLines-1) ? -1 : BCellToOctree[i + 1];
            int me = BCellToOctree[i];
            //If my left neighbor doesn't go to the same octnode I go to
            if (leftNeighbor != me) {
                //Then I am the first BCell/Facet belonging to my octnode
                facetPairs[me].first = i;
            }
            //If my right neighbor doesn't go the the same octnode I go to
            if (rightNeighbor != me) {
                //Then I am the last BCell/Facet belonging to my octnode
                facetPairs[me].last = i;
            }
        }
        return CL_SUCCESS;
    }

cl_int GetFacetPairs_p(cl::Buffer &orderedNodeIndices_i,
                       cl::Buffer &facetPairs_o, cl_int numLines,
                       cl_int octreeSize) {
  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("Kernels.GetFacetPairs_p");

  cl::CommandQueue *queue = &CLFW::DefaultQueue;
  cl::Kernel *kernel = &CLFW::Kernels["GetFacetPairsKernel"];
  int roundNumber = nextPow2(octreeSize);
  cl_int error = CLFW::get(facetPairs_o, "facetPairs", sizeof(FacetPair)* (roundNumber));
  FacetPair initialPair = { -1, -1 };
  error |= queue->enqueueFillBuffer<FacetPair>(facetPairs_o, { initialPair }, 0, sizeof(FacetPair) * roundNumber);
  error |= kernel->setArg(0, orderedNodeIndices_i);
  error |= kernel->setArg(1, facetPairs_o);
  error |= kernel->setArg(2, numLines);
  error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(numLines), cl::NullRange);

  if (logger.isEnabledFor(log4cplus::TRACE_LOG_LEVEL)) {
    vector<FacetPair> buf;
    DownloadFacetPairs(buf, facetPairs_o, octreeSize);
    LOG4CPLUS_TRACE(logger, "FacetPairs (" << octreeSize << ")");
    for (int i = 0; i < octreeSize; i++) {
      LOG4CPLUS_TRACE(logger, buf[i].first << " " << buf[i].last);
    }
  }

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

    floatn getNodeCenterFromLCP(BigUnsigned *LCP, unsigned int LCPLength, float octreeWidth) {
        unsigned int level = LCPLength / DIM;
        unsigned int lcpShift = LCPLength % DIM;
        floatn center = { 0.0, 0.0 };
        float centerShift = octreeWidth / (2.0 * (1 << level));

        for (int i = 0; i < level; ++i) {
            unsigned quadrant = getQuadrant(LCP, lcpShift, i);
            center.x += (quadrant & (1 << 0)) ? centerShift : -centerShift;
            center.y += (quadrant & (1 << 1)) ? centerShift : -centerShift;
#if DIM == 3
            center.z += (quadrant & (1 << 2)) ? centerShift : -centerShift;
#endif
            centerShift *= 2.0;
        }
        return center;
    }

    cl_int FindConflictCells_s(OctNode *octree, FacetPair *facetPairs, OctreeData *od, Conflict *conflicts,
        int* nodeToFacet, Line *lines, cl_int numLines, intn* points) {
        //Two lines are required for an ambigous cell to appear.
        if (numLines < 2) return CL_INVALID_ARG_SIZE;

        for (unsigned int i = 0; i < od->size; ++i) {
            FindConflictCells(octree, facetPairs, od, conflicts,
                nodeToFacet, lines, numLines, points, i);
        }
        return CL_SUCCESS;
    }

    cl_int FindConflictCells_p(cl::Buffer &octree, cl::Buffer &facetPairs, OctreeData &od, cl::Buffer &conflicts, cl::Buffer &nodeToFacet, cl::Buffer &lines, cl_int numLines, cl::Buffer &points) {
        //Two lines are required for an ambigous cell to appear.
        if (numLines < 2) return CL_INVALID_ARG_SIZE;
        cl::CommandQueue &queue = CLFW::DefaultQueue;
        cl::Kernel &kernel = CLFW::Kernels["FindConflictCellsKernel"];

        cl::Buffer bCellToOctnodeBuffer, initialConflictsBuffer;

        cl_int error = 0;
        bool isOld;
        Conflict initialPair;
        initialPair.color = -1;
        initialPair.i[0] = -1;
        initialPair.i[1] = -1;
        initialPair.i2[0] = -1;
        initialPair.i2[1] = -1;

        error |= CLFW::get(conflicts, "conflicts", 4 * nextPow2(od.size) * sizeof(Conflict));
        error |= CLFW::get(initialConflictsBuffer, "initialConflicts", 4 * nextPow2(od.size) * sizeof(Conflict), isOld);
        if (!isOld) {
            error |= queue.enqueueFillBuffer<Conflict>(initialConflictsBuffer, { initialPair }, 0, 4 * nextPow2(od.size) * sizeof(Conflict));
        }
        error |= queue.enqueueCopyBuffer(initialConflictsBuffer, conflicts, 0, 0, 4 * nextPow2(od.size) * sizeof(Conflict));

        error |= kernel.setArg(0, octree);
        error |= kernel.setArg(1, facetPairs);
        error |= kernel.setArg(2, points);
        error |= kernel.setArg(3, lines); 
        error |= kernel.setArg(4, nodeToFacet);
        error |= kernel.setArg(5, conflicts); 
        error |= kernel.setArg(6, numLines); 
        error |= kernel.setArg(7, od);
        error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(od.size), cl::NullRange);

        return error;
    }

    cl_int SampleConflictCounts_s(unsigned int totalOctnodes, Conflict *conflicts, int *totalAdditionalPoints,
        vector<int> &counts, Line* orderedLines, intn* qPoints, vector<intn> &newPoints) {
        cout << "series" << endl;
        *totalAdditionalPoints = 0;
        //This is inefficient. We should only iterate over the conflict leaves, not all leaves. (reduce to find total conflicts)
        for (int i = 0; i < totalOctnodes * 4; ++i) {
            Conflict c = conflicts[i];
            int currentTotalPoints = 0;

            if (conflicts[i].color == -2)
            {
                ConflictInfo info;
                Line firstLine = orderedLines[c.i[0]];
                Line secondLine = orderedLines[c.i[1]];
                intn q1 = qPoints[firstLine.firstIndex];
                intn q2 = qPoints[firstLine.secondIndex];
                intn r1 = qPoints[secondLine.firstIndex];
                intn r2 = qPoints[secondLine.secondIndex];
                sample_conflict_count(&info, q1, q2, r1, r2, c.origin, c.width);

                const int n = info.num_samples;
                for (int i = 0; i < info.num_samples; ++i) {
                    floatn sample;
                    sample_conflict_kernel(i, &info, &sample);
                    newPoints.push_back(convert_intn(sample));
                }

                *totalAdditionalPoints += n;

                ////Bug here...
                //if (currentTotalPoints == 0) {
                //    printf("Origin: %d %d Width %d (%d %d) (%d %d) : (%d %d) (%d %d) \n", conflicts[i].origin.x, conflicts[i].origin.y,
                //        conflicts[i].width, qPoints[firstLine.firstIndex].x, qPoints[firstLine.firstIndex].y,
                //        qPoints[firstLine.secondIndex].x, qPoints[firstLine.secondIndex].y,
                //        qPoints[secondLine.firstIndex].x, qPoints[secondLine.firstIndex].y,
                //        qPoints[secondLine.secondIndex].x, qPoints[secondLine.secondIndex].y);
                //}
            }
            counts[i] = currentTotalPoints;
            //    cl_int color;
            //cl_int i[2];
            //cl_float i2[2];
            //cl_int width;
            //intn origin;
        }
        return CL_SUCCESS;
    }

    cl_int GetResolutionPointsInfo_p(unsigned int totalOctnodes, cl::Buffer &conflicts, cl::Buffer &orderedLines,
        cl::Buffer &qPoints, cl::Buffer &conflictInfoBuffer, cl::Buffer &resolutionCounts, cl::Buffer &predicates) {
        cl::CommandQueue &queue = CLFW::DefaultQueue;
        cl::Kernel &kernel = CLFW::Kernels["CountResolutionPointsKernel"];

        int globalSize = nextPow2(totalOctnodes * 4);

        //TODO: Change this. Currently, this buffer is way too large...
        cl_int error = CLFW::get(conflictInfoBuffer, "conflictInfoBuffer", globalSize * sizeof(ConflictInfo));
        error |= CLFW::get(resolutionCounts, "resolutionCounts", globalSize * sizeof(cl_int));
        error |= CLFW::get(predicates, "resolutionPredicates", globalSize * sizeof(cl_int));

        error |= kernel.setArg(0, conflicts);
        error |= kernel.setArg(1, orderedLines);
        error |= kernel.setArg(2, qPoints);
        error |= kernel.setArg(3, predicates);
        error |= kernel.setArg(4, conflictInfoBuffer);
        error |= kernel.setArg(5, resolutionCounts);
        error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(totalOctnodes * 4), cl::NullRange);
        return error;
    }

// cl_int GetResolutionPointsInfo_s(
//     unsigned int totalOctnodes,
//     cl::Buffer &conflicts,
//     cl::Buffer &orderedLines,
//     cl::Buffer &qPoints,
//     cl::Buffer &conflictInfoBuffer,
//     cl::Buffer &resolutionCounts,
//     cl::Buffer &predicates) {

cl_int GetResolutionPointsInfo_s(
    unsigned int totalOctnodes,
    Conflict* conflicts,
    Line* orderedLines,
    intn* qPoints,
    ConflictInfo* conflictInfoBuffer,
    unsigned int* resolutionCounts,
    int* predicates) {

  for (int i = 0; i < totalOctnodes*4; ++i) {
    Conflict& c = conflicts[i];
    ConflictInfo& info = conflictInfoBuffer[i];
    info.currentNode = i;
    if (c.color == -2) {
      Line firstLine = orderedLines[c.i[0]];
      Line secondLine = orderedLines[c.i[1]];
      intn q1 = qPoints[firstLine.firstIndex];
      intn q2 = qPoints[firstLine.secondIndex];
      intn r1 = qPoints[secondLine.firstIndex];
      intn r2 = qPoints[secondLine.secondIndex];
      sample_conflict_count(&info, q1, q2, r1, r2, c.origin, c.width);
    }
  }

  return CL_SUCCESS;
}

cl_int GetResolutionPoints_p(
    unsigned int totalOctnodes, unsigned int totalAdditionalPoints,
    cl::Buffer &conflicts, cl::Buffer &orderedLines, cl::Buffer &qPoints,
    cl::Buffer &conflictInfoBuffer, cl::Buffer &scannedCounts,
    cl::Buffer &predicates, cl::Buffer &resolutionPoints) {

  cl::CommandQueue &queue = CLFW::DefaultQueue;
  cl::Kernel &kernel = CLFW::Kernels["GetResolutionPointsKernel"];
  cl_int error = 0;

  error |= CLFW::get(resolutionPoints, "ResPts", nextPow2(totalAdditionalPoints) * sizeof(intn));
  error |= kernel.setArg(0, conflicts);
  error |= kernel.setArg(1, orderedLines);
  error |= kernel.setArg(2, qPoints);
  error |= kernel.setArg(3, predicates);
  error |= kernel.setArg(4, conflictInfoBuffer);
  error |= kernel.setArg(5, scannedCounts);
  error |= kernel.setArg(6, resolutionPoints);
  error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(totalOctnodes * 4), cl::NullRange);
  return error;
}
}

/* Hybrid Kernels */
namespace Kernels {
//Approx 4% of build time
cl_int UniqueSorted(cl::Buffer &input, cl_int &size) {
  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("Kernels.UniqueSorted");

  startBenchmark("UniqueSorted");
  int globalSize = nextPow2(size);
  cl_int error = 0;

  cl::Buffer predicate, address, intermediate, result;
  error = CLFW::get(predicate, "predicate", sizeof(cl_int)*(globalSize));
  error |= CLFW::get(address, "address", sizeof(cl_int)*(globalSize));
  error |= CLFW::get(result, "result", sizeof(BigUnsigned) * globalSize);

  error |= UniquePredicate(input, predicate, globalSize);
  error |= StreamScan_p(predicate, address, globalSize, "UniqueIntermediate");
  error |= BUSingleCompact(input, result, predicate, address, globalSize);

  input = result;

  error |= CLFW::DefaultQueue.enqueueReadBuffer(address, CL_TRUE, (sizeof(cl_int)*globalSize - (sizeof(cl_int))), sizeof(cl_int), &size);
  // off by one error
  size--;
  stopBenchmark();

  if (logger.isEnabledFor(log4cplus::TRACE_LOG_LEVEL)) {
    vector<BigUnsigned> buf;
    DownloadZPoints(buf, input, size);
    LOG4CPLUS_TRACE(logger, "UniqueZPoints (" << size << ")");
    for (int i = 0; i < size; i++) {
      LOG4CPLUS_TRACE(logger, "  " << buToString(buf[i]));
    }
  }
        

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
cl_int QuantizePoints_p(cl_uint numPoints, cl::Buffer &unqPoints, cl::Buffer &qPoints, const floatn minimum, const int reslnWidth, const float bbWidth) {
  cl_int error = 0;
  cl_int roundSize = nextPow2(numPoints);
  error |= CLFW::get(qPoints, "qPoints", sizeof(intn)*roundSize);
  cl::Kernel kernel = CLFW::Kernels["QuantizePointsKernel"];
  error |= kernel.setArg(0, qPoints);
  error |= kernel.setArg(1, unqPoints);
  error |= kernel.setArg(2, minimum);
  error |= kernel.setArg(3, reslnWidth);
  error |= kernel.setArg(4, bbWidth);
  startBenchmark("QuantizePoints_p");
  error |= CLFW::DefaultQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numPoints), cl::NullRange);
  stopBenchmark();
  return error;
}

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
  error |= CLFW::DefaultQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
  stopBenchmark();
  return error;
};

cl_int PointsToMorton_s(cl_int size, cl_int bits, intn* points, BigUnsigned* result) {
  startBenchmark("PointsToMorton_s");
  for (int gid = 0; gid < size; ++gid) {
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
