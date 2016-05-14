#include "CLWrapper.h"
//--PUBLIC--//
// Initializes an OpenCL wrapper.
//    This includes automatically selecting a platform, automatically selecting a device,
//    initialing a context, command queue, and all kernels in the kernel box.
CLWrapper::CLWrapper(size_t defaultGlobalSize, size_t defaultLocalSize)
{
  error = CL_SUCCESS;
  initPlatformIds();
  initDevices();
  initContext();
  initCommandQueue();
  initKernelBox();
  globalSize = defaultGlobalSize;
  localSize = defaultLocalSize;
}
CLWrapper::~CLWrapper()
{
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}
shared_ptr<Buffer> CLWrapper::createBuffer(size_t size) {
  return make_shared<Buffer>(size, context, queue);
}
bool CLWrapper::isBufferUsable(shared_ptr<Buffer> buffer, size_t expectedSizeInBytes) {
  if (buffer == nullptr)
    return false;
  else if (buffer->getSize() != expectedSizeInBytes)
    return false;
  else
    return true;
}


//--PRIVATE--//
void CLWrapper::checkError(cl_int error){
  if (error != CL_SUCCESS) {
    std::cerr << "CLWrapper: OpenCL call failed with error " << error << std::endl;
    //std::getchar();
    //std::exit;
  }
}
void CLWrapper::initPlatformIds(){
  //Here, we find what OpenCL platforms we can use (Intel, NVidia, etc), and save them to platformIds.
  platformIdCount = 0;
  clGetPlatformIDs(0, nullptr, &platformIdCount);

  if (platformIdCount == 0) {
    std::cerr << "CLWrapper: No OpenCL platform found" << std::endl;
    std::getchar();
    std::exit;
  }
  else if (verbose)
    std::cout << "CLWrapper: Found " << platformIdCount << " platform(s)" << std::endl;

  platformIds.resize(platformIdCount);
  clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

  if (verbose)
  for (cl_uint i = 0; i < platformIdCount; ++i)
    std::cout << "CLWrapper: \t (" << (i + 1) << ") : " << getPlatformName(platformIds[i]) << std::endl;
}
void CLWrapper::initDevices(){
  //Here, we're hunting for avalable devices with the last platform avalable. (assumes graphics cards appear later in the list.)
  deviceIdCount = 0;
  clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

  if (deviceIdCount == 0) {
    std::cerr << "CLWrapper: No OpenCL devices found" << std::endl;
    std::getchar();
    std::exit;
  }
  else if (verbose)
    std::cout << "CLWrapper: Found " << deviceIdCount << " device(s) for " + getPlatformName(platformIds[0]) << std::endl;

  deviceIds.resize(deviceIdCount);
  clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);

  if (verbose)
  for (cl_uint i = 0; i < deviceIdCount; ++i)
    std::cout << "CLWrapper: \t (" << (i + 1) << ") : " << getDeviceName(deviceIds[i]) << std::endl;
}
void CLWrapper::initContext(){
  //Here, we're creating a context using the last platform avalable. (assumes graphics cards appear later in the list.)
  if (verbose)
    std::cout << "CLWrapper: Creating a context: ";
  const cl_context_properties contextProperties[] = {
    CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[0]), 0, 0
  };

  context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), nullptr, nullptr, &error);

  if (error != CL_SUCCESS) {
    std::cerr << "CLWrapper: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
  else if (verbose)
    std::cout << "SUCCESS" << std::endl;
}
void CLWrapper::initCommandQueue() {
  //Here, we select the first device avalable to us, and use it for our kernel command queue.
  if (verbose)
    std::cout << "CLWrapper: Creating a command queue: ";

  queue = clCreateCommandQueue(context, deviceIds[0], 0, &error);
  if (error != CL_SUCCESS) {
    std::cerr << "CLWrapper: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
  else if (verbose)
    std::cout << "SUCCESS" << std::endl;
}
void CLWrapper::initKernelBox(){
  //These are the files sent to be built by OpenCL

  //These files need to be copied relative to the executable...
  std::vector<std::string> files;
  files.push_back("../opencl/vec_cl.h");
	files.push_back("../C/BigUnsigned.h");
  files.push_back("../C/BigUnsigned.c");
  files.push_back("../C/ParallelAlgorithms.h");
  files.push_back("../C/ParallelAlgorithms.c");
  files.push_back("../C/BrtNode.h");
  files.push_back("../C/BuildBRT.h");
  files.push_back("../C/BuildBRT.c");
  files.push_back("../OpenCL/dim.h"); 
  files.push_back("../OctNode.h");
  files.push_back("../C/BuildOctree.h");
  files.push_back("../C/BuildOctree.c");
  files.push_back("../C/z_order.h");
  files.push_back("../C/z_order.c");
  files.push_back("../opencl/Kernels/kernels.cl");
  kernelBox = new KernelBox(files, context, queue, deviceIdCount, deviceIds);
}

std::string CLWrapper::getPlatformName(cl_platform_id id)
{
  size_t size = 0;
  clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);

  std::string result;
  result.resize(size);
  clGetPlatformInfo(id, CL_PLATFORM_NAME, size,
    const_cast<char*> (result.data()), nullptr);
  return result;
}
std::string CLWrapper::getDeviceName(cl_device_id id)
{
  size_t size = 0;
  clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);

  std::string result;
  result.resize(size);
  clGetDeviceInfo(id, CL_DEVICE_NAME, size,
    const_cast<char*> (result.data()), nullptr);

  return result;
}