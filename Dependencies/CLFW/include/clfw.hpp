#pragma once
/*
	Nate VM
	Idaho State University
	CLFW wraps OpenCL 2.0 calls in a easy to use framework.

	TODO:
		Add OpenGL interop
		Add device queues
*/
#define OPENCL_SOURCES_PATH "./Sources/OpenCL/opencl_sources.txt"
#define CL_HPP_TARGET_OPENCL_VERSION 200

/* Included files */
#include "cl2.hpp"

#include <vector>
#include <string>
#include <unordered_map>


enum Vendor { Nvidia, Intel, UnknownPlatform };

class CLFW{

private:
	static void print(std::string s, bool forced = false);
	static cl_int loadFile(const char* name, char** buffer, long* length);

	static bool verbose;
	static cl_int error;

	static bool lastBufferOld;

public:
	static Vendor SelectedVendor;
  
	/* Member Variables */
	static cl::Platform DefaultPlatform;
	static cl::Device DefaultDevice;
	static cl::Context DefaultContext;
	static cl::CommandQueue DefaultQueue;
	static cl::CommandQueue SecondaryQueue;
	static cl::DeviceCommandQueue DeviceQueue;

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

	/* Initializers */
	static cl_int Initialize(bool _verbose = false, int chosenDevice = -1, std::string buildOptions = "", unsigned int numQueues = 1);

	/* Accessors */
	static cl_int getPlatforms(std::vector<cl::Platform> &Platforms);
	static cl_int getDevices(std::vector<cl::Device> &Devices, int deviceType = CL_DEVICE_TYPE_ALL);
	static cl_int getContext(cl::Context &context, const cl::Device &device = DefaultDevice);
	static cl_int getQueue(cl::CommandQueue &queue, const cl::Context &context = DefaultContext, const cl::Device &device = DefaultDevice);
	static cl_int getSources(cl::Program::Sources &sources, std::vector<std::string> &files);
	static cl_int getSources(cl::Program::Sources &sources);
	static cl_int getKernels(std::unordered_map<std::string, cl::Kernel> &Kernels, cl::Program &program = DefaultProgram);
  static cl_int getBuffer(cl::Buffer &buffer, std::string key, cl_ulong size, bool &old = lastBufferOld, bool resize = false, int flag = CL_MEM_READ_WRITE);
	static cl_int getBestDevice(cl::Device &device, int characteristic = CL_DEVICE_MAX_COMPUTE_UNITS);
	static cl_int queryDevice(cl::Device &device);
	static cl_int buildProgram(cl::Program &program, cl::Program::Sources &sources, cl::Context &context = DefaultContext, cl::Device &device = DefaultDevice, std::string options = "");
	template<typename T>
	static cl_int Download(cl::Buffer & buffer, cl_int size, std::vector<T>& out);
	template<typename T>
	static cl_int Download(cl::Buffer & buffer, cl_int offset, T & out);
	template<typename T>
	static cl_int Upload(const std::vector<T> &input, cl::Buffer &buffer);
	template<typename T>
	static cl_int Upload(const std::vector<T> &input, cl_int offset, cl::Buffer &buffer);
	template<typename T>
	static cl_int Upload(T &input, cl_int offset, cl::Buffer &buffer);

  inline static cl_int NextPow2(cl_int num) {
    return std::max((int)std::pow(2, std::ceil(std::log((int)num) / std::log((int)2))), 8);
  }
};

template<typename T>
cl_int CLFW::Download(cl::Buffer &buffer, cl_int size, std::vector<T> &output) {
	output.resize(size);
	return CLFW::DefaultQueue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(T)*size, output.data());
}

template<typename T>
cl_int CLFW::Download(cl::Buffer &buffer, cl_int offset, T &output) {
	return CLFW::DefaultQueue.enqueueReadBuffer(buffer, CL_TRUE, offset * sizeof(T), sizeof(T), &output);
}

template<typename T>
cl_int CLFW::Upload(const std::vector<T> &input, cl::Buffer &buffer) {
	return CLFW::DefaultQueue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(T) * input.size(), input.data());
}

template<typename T>
cl_int CLFW::Upload(const std::vector<T> &input, cl_int offset, cl::Buffer &buffer) {
	return CLFW::DefaultQueue.enqueueWriteBuffer(buffer, CL_TRUE, offset * sizeof(T), sizeof(T) * input.size(), input.data());
}

// Note, T must be a power of two
template<typename T>
cl_int CLFW::Upload(T &input, cl_int offset, cl::Buffer &buffer) {
	return CLFW::DefaultQueue.enqueueWriteBuffer(buffer, CL_TRUE, sizeof(T) * offset, sizeof(T), &input);
}

std::string get_cl_error_msg(cl_int error);

#define print_cl_error(error) { \
	std::string msg = get_cl_error_msg(error); \
	std::cout << __FILE__ << " " << __LINE__ << " OpenCL error: " << msg << std::endl; \
	error = CL_SUCCESS; \
}

#define assert_cl_error(error) { \
	if (error != CL_SUCCESS) { \
		std::string msg = get_cl_error_msg(error);  \
		std::cout << __FILE__ << " " << __LINE__ << " OpenCL error: " << msg << std::endl; \
		} \
		assert(error == CL_SUCCESS); \
}
