#include "KernelBox_.h"

namespace KernelBox {
  cl_program program;
  Buffers buffers;
  bool initialized;

  const vector<string> Files = {
    "./opencl/C/BigUnsigned.c",
    "./opencl/C/ParallelAlgorithms.c",
    "./opencl/C/BuildBRT.c",
    "./opencl/C/BuildOctree.c",
    "./opencl/Kernels/kernels.cl"
  };

#define kernel(name) {#name, cl_kernel()}
  unordered_map<string, cl_kernel> Kernels = {
    kernel(PointsToMortonKernel),
    kernel(BitPredicateKernel),
    kernel(UniquePredicateKernel),
    kernel(StreamScanKernel),
    kernel(BUCompactKernel),
    kernel(BUSingleCompactKernel),
    kernel(BuildBinaryRadixTreeKernel),
    kernel(ComputeLocalSplitsKernel),
    kernel(BRT2OctreeKernel_init),
    kernel(BRT2OctreeKernel)
  };
#undef kernel

  int nextPow2(int num) { return max((int)pow(2, ceil(log(num) / log(2))), 8); }
  bool isBufferUsable(shared_ptr<Buffer> buffer, size_t expectedSizeInBytes) {
    if (buffer == nullptr)
      return false;
    else if (buffer->getSize() < expectedSizeInBytes)
      return false;
    else
      return true;
  }
  cl_int createBuffer(shared_ptr<Buffer> &buffer, size_t size) {
    if (!CLFW::Context) return CL_INVALID_CONTEXT;
    if (!CLFW::Queues[0]) return CL_INVALID_COMMAND_QUEUE;
    if (size <= 0) return CL_INVALID_BUFFER_SIZE;
    buffer = make_shared<Buffer>(size, CLFW::Context, CLFW::Queues[0]);
    if (!buffer) return CL_INVALID_PROPERTY;
    return CL_SUCCESS;
  }

  cl_int Initialize() {
    if (!initialized) {
      cl_int error = 0;
      error |= BuildOpenCLProgram(Files);
      error |= CreateKernels(program);
      if (error == CL_SUCCESS)
        initialized = true;
      else {
        initialized == false;
        return error;
      }
    }
    return CL_SUCCESS;
  }

  /* Building program */
  string loadFile(const char* name)
  {
    ifstream in(name);
    string result(
      (istreambuf_iterator<char>(in)),
      istreambuf_iterator<char>());
    return result;
  }
  cl_program createProgram(const string& source, cl_context context, cl_int &error)
  {
    size_t lengths[1] = { source.size() };
    const char* sources[1] = { source.data() };
    cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &error);
    return program;
  }
  cl_int BuildOpenCLProgram(const vector<string> Files) {
    if (CLFW::verbose) cout << "Building OpenCL program... " << endl;

    string entireProgram;
    for (int i = 0; i < Files.size(); ++i) {
      if (CLFW::verbose) cout << "\tAdding : " << Files[i] << endl;
      entireProgram += loadFile(Files[i].c_str());
    }

    cl_int error;
    program = createProgram(entireProgram, CLFW::Context, error);
    if (error != CL_SUCCESS) return error;

    error = clBuildProgram(program, CLFW::Devices.size(), CLFW::Devices.data(), nullptr, nullptr, nullptr);

    if (error == CL_BUILD_PROGRAM_FAILURE) {
      // Determine the size of the log
      size_t log_size;
      clGetProgramBuildInfo(program, CLFW::Devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

      // Allocate memory for the log
      char *log = (char *)malloc(log_size);

      // Get the log
      clGetProgramBuildInfo(program, CLFW::Devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

      // Print the log
      printf("%s\n", log);
      return error;
    }
    return error;
  }
  bool IsProgramInitialized() {
    if (!program) return false;
    return true;
  }
  bool IsInitialized() {
    return initialized;
  }

  /* Initializing kernels */
  cl_int CreateKernels(cl_program program) {
    cl_int error;
    for (auto kernel : Kernels) {
      Kernels.at(kernel.first) = clCreateKernel(program, kernel.first.c_str(), &error);
      if (error != CL_SUCCESS) return error;
    }

    return CL_SUCCESS;
  }

  /* Kernel calls */
  cl_int PointsToMorton(cl_int size, cl_int bits) {
    int globalSize = nextPow2(size);
    if (!isBufferUsable(buffers.bigUnsignedInput, sizeof(BigUnsigned)* (globalSize)))
       createBuffer(buffers.bigUnsignedInput, sizeof(BigUnsigned)* (globalSize));

    const size_t globalWorkSize[] = { globalSize, 0, 0 };
    cl_int error = 0;
    
    if (!buffers.points) return CL_INVALID_MEM_OBJECT;
    if (!buffers.bigUnsignedInput) return CL_INVALID_MEM_OBJECT;
    cl_mem clBUs = buffers.bigUnsignedInput->getBuffer();
    cl_mem clpoints = buffers.points->getBuffer();
    cout << Kernels.at("PointsToMortonKernel") << endl;
    error |= clSetKernelArg(Kernels.at("PointsToMortonKernel"), 0, sizeof(cl_mem), &clBUs);
    error |= clSetKernelArg(Kernels.at("PointsToMortonKernel"), 1, sizeof(cl_mem), &clpoints);
    error |= clSetKernelArg(Kernels.at("PointsToMortonKernel"), 2, sizeof(cl_int), &size);
    error |= clSetKernelArg(Kernels.at("PointsToMortonKernel"), 3, sizeof(cl_int), &bits);
    error |= clEnqueueNDRangeKernel(CLFW::Queues[0], Kernels.at("PointsToMortonKernel"), 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
    return error;
  };

  /*void BitPredicate(cl_mem input, cl_mem predicate, int &index, unsigned char compared, size_t globalSize) {
    const size_t globalWorkSize[] = { globalSize, 0, 0 };
    cl_ulong i = index;
    unsigned char c = compared;
    error = clSetKernelArg(bitPredicateKernel, 0, sizeof(cl_mem), &input);
    error = clSetKernelArg(bitPredicateKernel, 1, sizeof(cl_mem), &predicate);
    error = clSetKernelArg(bitPredicateKernel, 2, sizeof(Index), &i);
    error = clSetKernelArg(bitPredicateKernel, 3, sizeof(unsigned char), &c);
    error = clEnqueueNDRangeKernel(queue, bitPredicateKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
      std::cerr << "KernelBox predication: OpenCL call failed with error " << error << std::endl;
      std::getchar();
      std::exit;
    }
  };
  void UniquePredicate(cl_mem input, cl_mem predicate, size_t globalSize) {
    const size_t globalWorkSize[] = { globalSize, 0, 0 };
    error = clSetKernelArg(uniquePredicateKernel, 0, sizeof(cl_mem), &input);
    if (error != CL_SUCCESS) {
      std::cerr << "KernelBox unique predication 1: OpenCL call failed with error " << error << std::endl;
      std::getchar();
      std::exit;
    }
    error = clSetKernelArg(uniquePredicateKernel, 1, sizeof(cl_mem), &predicate);
    if (error != CL_SUCCESS) {
      std::cerr << "KernelBox unique predication 2: OpenCL call failed with error " << error << std::endl;
      std::getchar();
      std::exit;
    }
    error = clEnqueueNDRangeKernel(queue, uniquePredicateKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
      std::cerr << "KernelBox unique predication: OpenCL call failed with error " << error << std::endl;
      std::getchar();
      std::exit;
    }
  }
  size_t GetSteamScanWorkGroupSize(size_t globalSize) {
    using namespace std;
    size_t localSize;
    clGetKernelWorkGroupInfo(scanKernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &localSize, NULL);
    return min((int)(localSize), (int)globalSize);
  }
  void StreamScan(cl_mem input, cl_mem intermediate, cl_mem intermediateCopy, cl_mem result, size_t globalSize) {
    size_t localSize = getSteamScanWorkGroupSize(globalSize);
    const size_t globalWorkSize[] = { globalSize, 0, 0 };
    const size_t localWorkSize[] = { localSize, 0, 0 };
    //std::cout << localSize << std::endl;

    int currentNumWorkgroups = (globalSize / localSize);
    clEnqueueCopyBuffer(queue, intermediateCopy, intermediate, 0, 0, sizeof(Index)* currentNumWorkgroups, 0, NULL, NULL);

    clSetKernelArg(scanKernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(scanKernel, 1, sizeof(cl_mem), &result);
    clSetKernelArg(scanKernel, 2, sizeof(cl_mem), &intermediate);
    clSetKernelArg(scanKernel, 3, localSize * sizeof(Index), NULL);
    clSetKernelArg(scanKernel, 4, localSize * sizeof(Index), NULL);

    error = clEnqueueNDRangeKernel(queue, scanKernel, 1, 0, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
      std::cerr << "KernelBox stream scan: OpenCL call failed with error " << error << std::endl;
      std::getchar();
      std::exit;
    }
    lastNumWorkgroups = currentNumWorkgroups;
  };
  void SingleCompact(cl_mem inputBuffer, cl_mem resultBuffer, cl_mem PBuffer, cl_mem ABuffer, size_t globalSize) {
    const size_t globalWorkSize[] = { globalSize, 0, 0 };
    clSetKernelArg(singleCompactKernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(singleCompactKernel, 1, sizeof(cl_mem), &resultBuffer);
    clSetKernelArg(singleCompactKernel, 2, sizeof(cl_mem), &PBuffer);
    clSetKernelArg(singleCompactKernel, 3, sizeof(cl_mem), &ABuffer);
    error = clEnqueueNDRangeKernel(queue, singleCompactKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
      std::cerr << "KernelBox double compaction: OpenCL call failed with error " << error << std::endl;
      std::getchar();
      std::exit;
    }
  }
  void DoubleCompact(cl_mem inputBuffer, cl_mem resultBuffer, cl_mem resultBufferCopy, cl_mem LPBuffer, cl_mem LABuffer, size_t globalSize) {
    const size_t globalWorkSize[] = { globalSize, 0, 0 };
    clSetKernelArg(doubleCompactKernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(doubleCompactKernel, 1, sizeof(cl_mem), &resultBuffer);
    clSetKernelArg(doubleCompactKernel, 2, sizeof(cl_mem), &LPBuffer);
    clSetKernelArg(doubleCompactKernel, 3, sizeof(cl_mem), &LABuffer);
    clSetKernelArg(doubleCompactKernel, 4, sizeof(Index), &globalSize);

    BigUnsigned zero;
    initBlkBU(&zero, 0);
    //clEnqueueFillBuffer(queue, resultBuffer, &zero, sizeof(BigUnsigned), 0, sizeof(BigUnsigned)* (globalSize), 0, NULL, NULL);
    clEnqueueCopyBuffer(queue, resultBufferCopy, resultBuffer, 0, 0, sizeof(BigUnsigned) * globalSize, 0, NULL, NULL);
    error = clEnqueueNDRangeKernel(queue, doubleCompactKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
      std::cerr << "KernelBox double compaction: OpenCL call failed with error " << error << std::endl;
      std::getchar();
      std::exit;
    }
  };
  void BuildBinaryRadixTree(cl_mem internalNodes, cl_mem leafNodes, cl_mem mpoints, cl_int mbits, cl_int n, size_t globalSize) {
    const size_t globalWorkSize[] = { globalSize, 0, 0 };
    clSetKernelArg(binaryRadixTreeKernel, 0, sizeof(cl_mem), &internalNodes);
    clSetKernelArg(binaryRadixTreeKernel, 1, sizeof(cl_mem), &leafNodes);
    clSetKernelArg(binaryRadixTreeKernel, 2, sizeof(cl_mem), &mpoints);
    clSetKernelArg(binaryRadixTreeKernel, 3, sizeof(cl_int), &mbits);
    clSetKernelArg(binaryRadixTreeKernel, 4, sizeof(cl_int), &n);

    error = clEnqueueNDRangeKernel(queue, binaryRadixTreeKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
      std::cerr << "KernelBox double compaction: OpenCL call failed with error " << error << std::endl;
      std::getchar();
      std::exit;
    }
  }
  void ComputeLocalSplits(cl_mem localSplits, cl_mem localSplitsCopy, cl_mem I, size_t size, size_t globalSize) {
    const size_t globalWorkSize[] = { globalSize, 0, 0 };
    clSetKernelArg(computeLocalSplitsKernel, 0, sizeof(cl_mem), &localSplits);
    clSetKernelArg(computeLocalSplitsKernel, 1, sizeof(cl_mem), &I);
    clSetKernelArg(computeLocalSplitsKernel, 2, sizeof(cl_mem), &size);

    clEnqueueCopyBuffer(queue, localSplitsCopy, localSplits, 0, 0, sizeof(unsigned int) * globalSize, 0, NULL, NULL);

    error = clEnqueueNDRangeKernel(queue, computeLocalSplitsKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
      std::cerr << "KernelBox double compaction: OpenCL call failed with error " << error << std::endl;
      std::getchar();
      std::exit;
    }
  }
  void BRT2Octree(cl_mem I, cl_mem octree, cl_mem local_splits, cl_mem prefix_sums, cl_int n, size_t globalSize) {
    const size_t globalWorkSize[] = { globalSize, 0, 0 };
    clSetKernelArg(brt2OctreeKernel, 0, sizeof(cl_mem), &I);
    clSetKernelArg(brt2OctreeKernel, 1, sizeof(cl_mem), &octree);
    clSetKernelArg(brt2OctreeKernel, 2, sizeof(cl_mem), &local_splits);
    clSetKernelArg(brt2OctreeKernel, 3, sizeof(cl_mem), &prefix_sums);
    clSetKernelArg(brt2OctreeKernel, 4, sizeof(cl_int), &n);

    error = clEnqueueNDRangeKernel(queue, brt2OctreeKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
      std::cerr << "KernelBox double compaction: OpenCL call failed with error " << error << std::endl;
      std::getchar();
      std::exit;
    }
  }
  void BRT2Octree_init(cl_mem I, cl_mem octree, cl_mem local_splits, cl_mem prefix_sums, cl_int n, size_t globalSize) {
    const size_t globalWorkSize[] = { globalSize, 0, 0 };
    clSetKernelArg(brt2OctreeKernel_init, 0, sizeof(cl_mem), &I);
    clSetKernelArg(brt2OctreeKernel_init, 1, sizeof(cl_mem), &octree);
    clSetKernelArg(brt2OctreeKernel_init, 2, sizeof(cl_mem), &local_splits);
    clSetKernelArg(brt2OctreeKernel_init, 3, sizeof(cl_mem), &prefix_sums);
    clSetKernelArg(brt2OctreeKernel_init, 4, sizeof(cl_int), &n);

    error = clEnqueueNDRangeKernel(queue, brt2OctreeKernel_init, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
      std::cerr << "KernelBox double compaction: OpenCL call failed with error " << error << std::endl;
      std::getchar();
      std::exit;
    }
  }*/
}