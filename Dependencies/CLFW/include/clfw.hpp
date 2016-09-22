#pragma once
/*
  Hi!

  This is the CLFW library! 
  
  CLFW wraps OpenCL calls in a easy to use framework.
*/

/* Included files */
#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include <CL/cl.hpp>
#endif

#include <vector>
#include <string>
#include <unordered_map>

class CLFW{

private:
  /* Verbose things */
  static const int errorFG = 1;
  static const int errorBG = 41;
  static const int successFG = 30;
  static const int successBG = 42;
  static const int infoFG = 30;
  static const int infoBG = 47;
  static const int defaultFG = 39;
  static const int defaultBG = 49;
  static void Print(std::string s, int fgcode = defaultFG, int bgcode = defaultBG, bool verbose = true);

  /* Source file management */
  static cl_int loadFile(const char* name, char** buffer, long* length);

public:
  static bool verbose;
  static bool lastBufferOld;
  
  /* Member Variables */
  static cl::Platform DefaultPlatform;
  static cl::Device DefaultDevice;
  static cl::Context DefaultContext;
  static cl::CommandQueue DefaultQueue;

  static cl::Program DefaultProgram;
  static cl::Program::Sources DefaultSources;

  /* Lists */
  static std::vector<cl::Platform> Platforms;
  static std::vector<cl::Device> Devices;
  static std::vector<cl::Context> Contexts;
  static std::vector<cl::CommandQueue> Queues;

  /* Maps*/
  static std::unordered_map<std::string, cl::Kernel> Kernels;
  static std::unordered_map<std::string, cl::Buffer> Buffers;

  /* Queries */
  static bool IsNotInitialized();

  /* Initializers */
  static cl_int Initialize(bool _verbose = false, bool queryMode = false, unsigned int numQueues = 1);

  /* Accessors */
  static cl_int get(std::vector<cl::Platform> &Platforms);
  static cl_int get(std::vector<cl::Device> &Devices, int deviceType = CL_DEVICE_TYPE_ALL);
  static cl_int get(cl::Context &context, const cl::Device &device = DefaultDevice);
  static cl_int get(cl::CommandQueue &queue, const cl::Context &context = DefaultContext, const cl::Device &device = DefaultDevice);
  static cl_int get(cl::Program::Sources &sources, std::vector<std::string> &files);
  static cl_int get(cl::Program::Sources &sources);
  static cl_int get(std::unordered_map<std::string, cl::Kernel> &Kernels, cl::Program &program = DefaultProgram);
  static cl_int get(cl::Buffer &buffer, std::string key, cl_ulong size, bool &old = lastBufferOld, cl::Context &context = DefaultContext, int flag = CL_MEM_READ_WRITE);
  static cl_int getBest(cl::Device &device, int characteristic = CL_DEVICE_MAX_COMPUTE_UNITS);
  static cl_int query(cl::Device &device);
  static cl_int Build(cl::Program &program, cl::Program::Sources &sources, cl::Context &context = DefaultContext, cl::Device &device = DefaultDevice);
};
