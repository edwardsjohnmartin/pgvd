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
}