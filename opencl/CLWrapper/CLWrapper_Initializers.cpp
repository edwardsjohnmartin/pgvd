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

  printInfo(deviceIds[0]);
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
  else if (buffer->getSize() < expectedSizeInBytes)
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
  clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceIdCount);

  if (deviceIdCount == 0) {
    std::cerr << "CLWrapper: No OpenCL devices found" << std::endl;
    std::getchar();
    std::exit;
  }
  else if (verbose)
    std::cout << "CLWrapper: Found " << deviceIdCount << " device(s) for " + getPlatformName(platformIds[0]) << std::endl;

  deviceIds.resize(deviceIdCount);
  clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_GPU, deviceIdCount, deviceIds.data(), nullptr);

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

//----------------------------------------
// printInfoT
//----------------------------------------
template <typename T>
void printInfoT(
    const cl_device_id id, const cl_device_info param_name,
    const std::string& name) {

  size_t size = 0;
  clGetDeviceInfo(id, param_name, 0, nullptr, &size);

  T result;
  clGetDeviceInfo(id, param_name, size, &result, nullptr);
  std::cout << name << ": " << result << endl;
}

//----------------------------------------
// printInfoT - char[] version
//----------------------------------------
template <>
void printInfoT<char[]>(
    const cl_device_id id, const cl_device_info param_name,
    const std::string& name) {

  size_t size = 0;
  clGetDeviceInfo(id, param_name, 0, nullptr, &size);

  std::string result;
  result.resize(size);
  clGetDeviceInfo(id, param_name, size,
    const_cast<char*> (result.data()), nullptr);
  std::cout << name << ": " << result << endl;
}

//----------------------------------------
// printInfoT - size_t[] version
//----------------------------------------
template <>
void printInfoT<size_t[]>(
    const cl_device_id id, const cl_device_info param_name,
    const std::string& name) {

  size_t size = 0;
  clGetDeviceInfo(id, param_name, 0, nullptr, &size);

  const int n = size / sizeof(size_t);
  size_t* result = new size_t[n];

  clGetDeviceInfo(id, param_name, size, result, nullptr);
  std::cout << name << ": ";
  for (int i = 0; i < n; ++i) {
    cout << result[i] << " ";
  }
  cout << endl;

  delete [] result;
}

void CLWrapper::printInfo(const cl_device_id id) const {
  printInfoT<cl_uint>(
      id, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
  printInfoT<cl_bool>(
      id, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
  printInfoT<cl_bool>(
      id, CL_DEVICE_COMPILER_AVAILABLE, "CL_DEVICE_COMPILER_AVAILABLE");
  printInfoT<cl_device_fp_config>(
      id, CL_DEVICE_DOUBLE_FP_CONFIG, "CL_DEVICE_DOUBLE_FP_CONFIG");
  printInfoT<cl_bool>(
      id, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
  printInfoT<cl_bool>(
      id, CL_DEVICE_ERROR_CORRECTION_SUPPORT,
      "CL_DEVICE_ERROR_CORRECTION_SUPPORT");
  printInfoT<cl_device_exec_capabilities>(
      id, CL_DEVICE_EXECUTION_CAPABILITIES, "CL_DEVICE_EXECUTION_CAPABILITIES");
  printInfoT<char[]>(
      id, CL_DEVICE_EXTENSIONS, "CL_DEVICE_EXTENSIONS");
  printInfoT<cl_ulong>(
      id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
  printInfoT<cl_device_mem_cache_type>(
      id, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE");
  printInfoT<cl_uint>(
      id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
      "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
  printInfoT<cl_ulong>(
      id, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
  printInfoT<cl_device_fp_config>(
      id, CL_DEVICE_HALF_FP_CONFIG, "CL_DEVICE_HALF_FP_CONFIG");
  printInfoT<cl_bool>(
      id, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
  printInfoT<size_t>(
      id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, "CL_DEVICE_IMAGE2D_MAX_HEIGHT");
  printInfoT<size_t>(
      id, CL_DEVICE_IMAGE2D_MAX_WIDTH, "CL_DEVICE_IMAGE2D_MAX_WIDTH");
  printInfoT<size_t>(
      id, CL_DEVICE_IMAGE3D_MAX_DEPTH, "CL_DEVICE_IMAGE3D_MAX_DEPTH");
  printInfoT<size_t>(
      id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, "CL_DEVICE_IMAGE3D_MAX_HEIGHT");
  printInfoT<size_t>(
      id, CL_DEVICE_IMAGE3D_MAX_WIDTH, "CL_DEVICE_IMAGE3D_MAX_WIDTH");
  printInfoT<cl_ulong>(
      id, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
  printInfoT<cl_device_local_mem_type>(
      id, CL_DEVICE_LOCAL_MEM_TYPE, "CL_DEVICE_LOCAL_MEM_TYPE");
  printInfoT<cl_uint>(
      id, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
  printInfoT<cl_uint>(
      id, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
  printInfoT<cl_uint>(
      id, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
  printInfoT<cl_ulong>(
      id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
      "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
  printInfoT<cl_ulong>(
      id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, "CL_DEVICE_MAX_MEM_ALLOC_SIZE");
  printInfoT<size_t>(
      id, CL_DEVICE_MAX_PARAMETER_SIZE, "CL_DEVICE_MAX_PARAMETER_SIZE");
  printInfoT<cl_uint>(
      id, CL_DEVICE_MAX_READ_IMAGE_ARGS, "CL_DEVICE_MAX_READ_IMAGE_ARGS");
  printInfoT<cl_uint>(
      id, CL_DEVICE_MAX_SAMPLERS, "CL_DEVICE_MAX_SAMPLERS");
  printInfoT<size_t>(
      id, CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE");
  printInfoT<cl_uint>(
      id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
      "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
  printInfoT<size_t[]>(
      id, CL_DEVICE_MAX_WORK_ITEM_SIZES, "CL_DEVICE_MAX_WORK_ITEM_SIZES");
  printInfoT<cl_uint>(
      id, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, "CL_DEVICE_MAX_WRITE_IMAGE_ARGS");
  printInfoT<cl_uint>(
      id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
  printInfoT<cl_uint>(
      id, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
      "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
  printInfoT<char[]>(
      id, CL_DEVICE_NAME, "CL_DEVICE_NAME");
  printInfoT<cl_platform_id>(
      id, CL_DEVICE_PLATFORM, "CL_DEVICE_PLATFORM");
  printInfoT<char[]>(
      id, CL_DEVICE_PROFILE, "CL_DEVICE_PROFILE");
  printInfoT<size_t>(
      id, CL_DEVICE_PROFILING_TIMER_RESOLUTION,
      "CL_DEVICE_PROFILING_TIMER_RESOLUTION");
  printInfoT<cl_command_queue_properties>(
      id, CL_DEVICE_QUEUE_PROPERTIES, "CL_DEVICE_QUEUE_PROPERTIES");
  printInfoT<cl_device_type>(
      id, CL_DEVICE_TYPE, "CL_DEVICE_TYPE");
  printInfoT<char[]>(
      id, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
  printInfoT<cl_uint>(
      id, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
  printInfoT<char[]>(
      id, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
}
