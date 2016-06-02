#include <iostream>
#include "clfw.hpp"
using namespace std;

/* Flags */
bool CLFW::verbose;

/* Member Variables */
cl_context CLFW::Context;

/* Member Lists */
vector<cl_platform_id> CLFW::Platforms;
vector<cl_device_id> CLFW::Devices;
vector<cl_command_queue> CLFW::Queues;

/* Queries */
string CLFW::GetPlatformName(cl_platform_id id)
{
  // Get the number of characters in the platform.
  size_t size = 0;
  clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);

  // Get the platform name.
  string result;
  result.resize(size);
  clGetPlatformInfo(id, CL_PLATFORM_NAME, size, const_cast<char*> (result.data()), nullptr);
  return result;
}
string CLFW::GetDeviceName(cl_device_id id)
{
  // Get the number of characters in the device name.
  size_t size = 0;
  clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);

  // Get the device name.
  string result;
  result.resize(size);
  clGetDeviceInfo(id, CL_DEVICE_NAME, size,
    const_cast<char*> (result.data()), nullptr);

  return result;
}
bool CLFW::IsInitialized() {
  if (!Context) return false;
  if (Queues.size() == 0) return false;
  return true;
}

/* Initializers */
cl_int CLFW::Initialize(bool _verbose) {
  //Free old queues and contexts.
  if (Queues.size() > 0 || Context != NULL) {
    verbose = false;
    CLFW::Terminate();
  }
  verbose = _verbose;
  if (verbose) cout << "CLFW: Initializing..." << endl;
  cl_int error = 0;
  error |= CLFW::InitializePlatformList();
  error |= CLFW::InitializeDeviceList(0);
  error |= CLFW::InitializeContext(0);
  error |= CLFW::AddQueue(0);
  return error;
}
cl_int CLFW::InitializePlatformList() {
  // Query how many platforms are available.
  cl_uint platformIdCount = 0;
  cl_int error;
  error = clGetPlatformIDs(0, nullptr, &platformIdCount);
  if (error != CL_SUCCESS) return error;

  if (verbose) cout << "CLFW: Found " << platformIdCount << " platform(s)" << endl;

  Platforms.resize(platformIdCount);
  error = clGetPlatformIDs(platformIdCount, Platforms.data(), nullptr);
  if (error != CL_SUCCESS) return error;

  if (verbose)
    for (cl_uint i = 0; i < platformIdCount; ++i)
      cout << "CLFW: [" << i << "] -> " << GetPlatformName(Platforms[i]) << endl;
  return CL_SUCCESS;
}
cl_int CLFW::InitializeDeviceList(int platformIndex, int deviceType) {
  if (platformIndex > Platforms.size() - 1) return CL_INVALID_PLATFORM;
  
  //Query how many of the given device type there are.
  cl_uint deviceIdCount = 0;
  cl_int error;
  error = clGetDeviceIDs(Platforms[platformIndex], deviceType, 0, nullptr, &deviceIdCount);
  if (error != CL_SUCCESS) return error;
  
  if (verbose) cout << "CLFW: Found " << deviceIdCount << " device(s) for " + GetPlatformName(Platforms[platformIndex]) << endl;

  Devices.resize(deviceIdCount);
  clGetDeviceIDs(Platforms[platformIndex], deviceType, deviceIdCount, Devices.data(), nullptr);

  if (verbose)
    for (cl_uint i = 0; i < deviceIdCount; ++i)
      cout << "CLFW: [" << i << "] : " << GetDeviceName(Devices[i]) << endl;
  return CL_SUCCESS;
}
cl_int CLFW::InitializeContext(int platformIndex) {
  if (platformIndex > Platforms.size() - 1) return CL_INVALID_PLATFORM;
  if (verbose) cout << "CLFW: Creating a context for "
    << GetPlatformName(Platforms[platformIndex]) << endl;

  const cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (Platforms[platformIndex]), 0, 0};

  cl_int error;
  Context = clCreateContext(contextProperties, Devices.size(), Devices.data(), nullptr, nullptr, &error);

  return error;
}

cl_int CLFW::AddQueue(int deviceIndex) {
  if (deviceIndex > Devices.size() - 1) return CL_INVALID_DEVICE;

  //Here, we select the first device avalable to us, and use it for our kernel command queue.
  if (verbose) cout << "CLFW: Creating a command queue: ";
  cl_int error;
  cl_command_queue queue = clCreateCommandQueue(Context, Devices[deviceIndex], 0, &error);
  if (error == CL_SUCCESS) Queues.push_back(queue);
  return error;
}

cl_int CLFW::Terminate() {
  if (verbose) cout << "CLFW: Terminating..." << endl;
  for (int i = 0; i < Queues.size(); ++i) {
    clReleaseCommandQueue(Queues[i]);
  }
  Queues.resize(0);
  clReleaseContext(Context);
  return CL_SUCCESS;
}

