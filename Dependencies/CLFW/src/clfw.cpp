#include <iostream>
#include "clfw.hpp"

#include "../../../Sources/Options/Options.h"
//using namespace std;
//using namespace cl;
/* Verbose things */
bool CLFW::verbose = true;
bool CLFW::lastBufferOld = false;
#if defined(_MSC_VER)
#include <Windows.h>
#include <stdio.h>BOOL isDebuggerPresent();
#endif

void CLFW::Print(std::string s, int fgcode, int bgcode, bool _verbose) {
//#ifdef _MSC_VER
//  if (IsDebuggerPresent()) {
    if (_verbose || ((fgcode == errorFG) && (bgcode = errorBG)))
      printf("CLFW: %-70s\n", s.c_str());
//  }
//  else {
//    if (_verbose || ((fgcode == errorFG) && (bgcode = errorBG))) {
//      std::cout << '\033';
//      printf("%c[%d;%dmCLFW: %-70s%c[0m\n", '\033', fgcode, bgcode, s.c_str(), '\033');
//      std::cout << '\033'<< "0m";
//
//    }
//  }
//#else
//  if (_verbose || ((fgcode == errorFG) && (bgcode = errorBG))) 
//    printf("\033[%d;%dmCLFW: %-70s\033[0m\n",fgcode, bgcode, s.c_str());
//#endif // _MSC_VER
}

/* Source file management */
cl_int CLFW::loadFile(const char* name, char** buffer, long* length)
{
  FILE * f = fopen(name, "rb");
  if (f)
  {
	  fseek(f, 0, SEEK_END);
	  *length = ftell(f);
	  fseek(f, 0, SEEK_SET);
	  *buffer = (char*)malloc(*length + 1);
	  
	  if (*buffer)
	  {
		  fread(*buffer, 1, *length, f);
		  //fgets(*buffer, *length+1, f);
	  }
	  fclose(f);
	  (*buffer)[*length] = '\0';
	  }
  else
	  return CL_INVALID_VALUE;
  return CL_SUCCESS;
}

/* Member Variables */
Vendor CLFW::SelectedVendor;
cl::Device CLFW::DefaultDevice;
cl::Context CLFW::DefaultContext;
cl::CommandQueue CLFW::DefaultQueue;
cl::CommandQueue CLFW::SecondaryQueue;

cl::Program CLFW::DefaultProgram;
cl::Program::Sources CLFW::DefaultSources;

/* Lists */
std::vector<cl::Platform> CLFW::Platforms;
std::vector<cl::Device> CLFW::Devices;
std::vector<cl::Context> CLFW::Contexts;
std::vector<cl::CommandQueue> CLFW::Queues;

/* Maps */
std::unordered_map<std::string, cl::Kernel> CLFW::Kernels;
std::unordered_map<std::string, cl::Buffer> CLFW::Buffers;

/* Queries */
bool CLFW::IsNotInitialized() {
  if (Platforms.size() == 0) return true;
  if (Devices.size() == 0) return true;
  if (Contexts.size() == 0) return true;
  if (Queues.size() == 0) return true;
  return false;
}

/* Initializers */
cl_int CLFW::Initialize(bool _verbose, bool queryMode, unsigned int numQueues) {
  verbose = _verbose;
  if (verbose) Print("Initializing...", infoFG, infoBG);
  cl_int  error = get(Platforms);
  error |= get(Devices);

  if (queryMode == false) {
    error |= getBest(DefaultDevice);
  }
  else {
    error |= query(DefaultDevice);
  }
  Contexts.clear();
  error |= get(DefaultContext);
  Contexts.push_back(DefaultContext);
  Queues.clear();
  for (int i = 0; i < numQueues; ++i) {
    cl::CommandQueue queue;
    error |= get(queue);
    Queues.push_back(queue);
    if (i == 0)
      DefaultQueue = queue;
    if (i == 1)
      SecondaryQueue = queue;
  }
  error |= get(DefaultSources);

  error |= Build(DefaultProgram, DefaultSources);
  error |= get(Kernels);
  return error;
}

/* Accessors */
cl_int CLFW::get(std::vector<cl::Platform> &Platforms) {
  cl_int error = cl::Platform::get(&Platforms);
  if (Platforms.size() == 0) {
    Print("No platforms found. Check OpenCL installation!", errorFG, errorBG);
    return error;
  }

  Print("Found " + std::to_string(Platforms.size()) + " platforms(s)", successFG, successBG);
  for (cl_uint i = 0; i < Platforms.size(); ++i)
    Print("[" + std::to_string(i) + "] -> " + Platforms[i].getInfo<CL_PLATFORM_NAME>(), infoFG, infoBG);
  
  return CL_SUCCESS;
}
cl_int CLFW::get(std::vector<cl::Device> &Devices, int deviceType) {
  if (Platforms.size() == 0) {
    Print("No platforms found. Check OpenCL installation!", errorFG, errorBG);
    return CL_INVALID_VALUE;
  }
  
  Devices.resize(0);
  
  cl_int error = 0;
  std::vector<cl::Device> temp;
  for (int i = 0; i < Platforms.size(); ++i) {
    temp.clear();
    error |= Platforms[i].getDevices(deviceType, &temp);
    Devices.insert(Devices.end(), temp.begin(), temp.end());
  }

  if (Devices.size() == 0) {
    Print("No devices found. Check OpenCL installation!", errorFG, errorBG);
    return error;
  }

  if (verbose)
    Print("Found " + std::to_string(Devices.size()) + " device(s) for " + std::to_string(Platforms.size()) + " platform(s)", successFG, successBG);
  for (cl_uint i = 0; i < Devices.size(); ++i)
    Print("[" + std::to_string(i) + "] : " + Devices[i].getInfo<CL_DEVICE_NAME>(), infoFG, infoBG);

  return CL_SUCCESS;
}
cl_int CLFW::get(cl::Context &context, const cl::Device &device) {
  cl_int error = 0;
  context = cl::Context({ device }, NULL, NULL, NULL, &error);
  if (error == CL_SUCCESS) Print("Created context for " + device.getInfo<CL_DEVICE_NAME>(), successFG, successBG);
  else Print("Failed creating context for " + device.getInfo<CL_DEVICE_NAME>(), errorFG, errorBG);
  return error;
}
cl_int CLFW::get(cl::CommandQueue &queue, const cl::Context &context, const cl::Device &device) {
  cl_int error = 0;
  queue = cl::CommandQueue(context, device, error);
  if (error == CL_SUCCESS) Print("Created queue for DefaultContext's " + device.getInfo<CL_DEVICE_NAME>(), successFG, successBG);
  else Print("Failed creating queue for DefaultContext's " + device.getInfo<CL_DEVICE_NAME>(), errorFG, errorBG);
  return error;
}
cl_int CLFW::get(cl::Program::Sources &sources, std::vector<std::string> &files) {
  sources.clear();
  cl_int error = 0;
  if (files.size() == 0) return CL_INVALID_VALUE;
  for (int i = 0; i < files.size(); ++i) {
    Print("adding " + files[i] + " to sources.", infoFG, infoBG);
    long length = 0;
    char *source = 0;
    error|=loadFile(files[i].c_str(), &source, &length);
	sources.push_back({ source, length });
  }
  return error;
}
cl_int CLFW::get(cl::Program::Sources &sources) {
  sources.clear();
  char* text = 0;
  long temp;
  loadFile("./Sources/OpenCL/opencl_sources.txt", &text, &temp);
  try {
    char* file = strtok(text, "\n\r\n\0");
    do {
      Print("adding " + std::string(file) + " to sources.", infoFG, infoBG);
      long length = 0;
      char *source = 0;
      loadFile(file, &source, &length);
      sources.push_back({ source, length });
      file = strtok(NULL, "\n\r\n\0");
    } while (file != NULL);
  }
  catch (...) {
    Print("Unable to open ./Sources/OpenCL/opencl_sources.txt!!", errorFG, errorBG);
  }
  return CL_SUCCESS;
}
cl_int CLFW::get(std::unordered_map<cl::STRING_CLASS, cl::Kernel> &Kernels, cl::Program &program) {
  Kernels.clear();
  std::vector<cl::Kernel> tempKernels;
  cl_int error = program.createKernels(&tempKernels);
  if (error != CL_SUCCESS) {
    Print("Unable to create kernels.", errorFG, errorBG);
    return error;
  }
  for (int i = 0; i < tempKernels.size(); ++i) {
	  std::string temp = std::string(tempKernels[i].getInfo<CL_KERNEL_FUNCTION_NAME>());
    
    //For some reason, OpenCL string's lengths are 1 char longer than they should be.
    temp = temp.substr(0, temp.length() - 1);
    Kernels[temp] = tempKernels[i];
  }
  for (auto i : Kernels) {
    Print("Created Kernel " + i.first, successFG, successBG);
  }
  return CL_SUCCESS;
}
cl_int CLFW::get(cl::Buffer &buffer, std::string key, cl_ulong size, bool &old, cl::Context &context, int flag) {
  cl_int error = 0;
  old = true;
  //If the key is not found...
  if (Buffers.find(key) == Buffers.end()) old = false;
  else if (Buffers[key].getInfo<CL_MEM_SIZE>() != size) old = false;

  if (old == false) 
  {
    Buffers[key] = cl::Buffer(context, flag, size, NULL, &error);
    if (error == CL_SUCCESS)
      Print("Created buffer " + key + " of size " + std::to_string(size) + " bytes" , successFG, successBG);
    else
      Print("Failed to create buffer " + key + " of size " + std::to_string(size) + " bytes. Error #  %d", successFG, successBG, error);
  }
  buffer = Buffers[key];
  return error;
}

cl_int CLFW::getBest(cl::Device &device, int characteristic) {
  cl_int error;
  int largest = 0;
  int temp;

  const int COMBINED = CL_DEVICE_MAX_CLOCK_FREQUENCY & CL_DEVICE_MAX_COMPUTE_UNITS & CL_DEVICE_MAX_WORK_GROUP_SIZE;
  
  if (characteristic != CL_DEVICE_MAX_CLOCK_FREQUENCY &&
      characteristic != CL_DEVICE_MAX_COMPUTE_UNITS &&
	  characteristic != CL_DEVICE_MAX_WORK_GROUP_SIZE &&
	  characteristic != COMBINED)
  {
    Print("Device characteristic unrecognized! ", errorFG, errorBG);
    return CL_INVALID_VALUE;
  }

  for (int i = 0; i < Devices.size(); i++) {
    switch (characteristic) {
      case CL_DEVICE_MAX_COMPUTE_UNITS:
        temp = Devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        break;
      case CL_DEVICE_MAX_CLOCK_FREQUENCY:
        temp = Devices[i].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
        break;
	  case CL_DEVICE_MAX_WORK_GROUP_SIZE:
		  temp = Devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		  break;
	  case COMBINED:
		  temp = Devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * Devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() * Devices[i].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
    }
    if (largest < temp) {
      largest = temp;
      device = Devices[i];
    }
  }
  Print("Selected " + device.getInfo<CL_DEVICE_NAME>(), successFG, successBG);
  return CL_SUCCESS;
}

cl_int CLFW::query(cl::Device &device) {
  cl_int error;
  if (Devices.size() == 0) return CL_INVALID_ARG_SIZE;

  Print("Which device would you like to use? (enter a number between 0 and " + std::to_string(Devices.size()-1) + ")", infoFG, infoBG, true);

  for (cl_uint i = 0; i < Devices.size(); ++i) {
    Print("[" + std::to_string(i) + "] : " + Devices[i].getInfo<CL_DEVICE_NAME>() 
      + " " + Devices[i].getInfo<CL_DEVICE_VERSION>().c_str(), infoFG, infoBG, true);
    Print(Devices[i].getInfo<CL_DEVICE_VERSION>(), infoFG, infoBG, true);
  }


  int selection = Options::device;
  if (selection == -1) {
    do {
      while (!(std::cin >> selection)) {
        std::cin.clear();
        while (std::cin.get() != '\n') continue;
      }
    } while (selection >= Devices.size());
  }

  device = Devices[selection];
  Print("Selected " + device.getInfo<CL_DEVICE_NAME>(), successFG, successBG, true);
  cl_platform_id selectedPlatformId = device.getInfo<CL_DEVICE_PLATFORM>();
  cl::Platform platform(selectedPlatformId);
  std::string vendor = platform.getInfo<CL_PLATFORM_VENDOR>();
  if (vendor.find("NVIDIA") != std::string::npos) {
    SelectedVendor = Vendor::Nvidia;
  }
  else if (vendor.find("Intel") != std::string::npos) {
    SelectedVendor = Vendor::Intel;
  } 
  else SelectedVendor = Vendor::UnknownPlatform;
  return CL_SUCCESS;
}

cl_int CLFW::Build(cl::Program &program, cl::Program::Sources &sources, cl::Context &context, cl::Device &device) {
  cl_int error;
  //char* spir;
  //long length;
  //cl_int spirError;
  //loadFile("./kernel.spir", &spir, (long*)&length);
  //cl_program spirProgram = clCreateProgramWithIL(context(), spir, length, &error);

  //program = cl::Program(spirProgram);
  program = cl::Program(context, sources, &error);


  if (error != CL_SUCCESS) {
	  Print("Error creating program:", errorFG, errorBG);
	  return error;
  }

  std::cout << "Building cl program with options \"" << Options::cl_options << "\"" << std::endl;
  error = program.build({ device }, Options::cl_options.data());
  if (error != CL_SUCCESS) {
    Print("Error building program:", errorFG, errorBG);
    Print(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device), errorFG, errorBG);
  }
  else {
    Print("Success building OpenCL program. ", successFG, successBG);
  }
  
  return error;
}

std::string get_cl_error_msg(cl_int error) {
  std::string msg;
  switch (error) {
    case CL_INVALID_PROGRAM_EXECUTABLE:
      msg = "if there is no successfully built program executable available for device associated with command_queue.";
      break;
    case CL_INVALID_COMMAND_QUEUE:
      msg = "if command_queue is not a valid command-queue.";
      break;
    case CL_INVALID_KERNEL:
      msg = "if kernel is not a valid kernel object.";
      break;
    case CL_INVALID_CONTEXT:
      msg = "if context associated with command_queue and kernel is not the same or if the context associated with command_queue and events in event_wait_list are not the same.";
      break;
    case CL_INVALID_KERNEL_ARGS:
      msg = "if the kernel argument values have not been specified.";
      break;
    case CL_INVALID_WORK_DIMENSION:
      msg = "if work_dim is not a valid value (i.e. a value between 1 and 3).";
      break;
    case CL_INVALID_GLOBAL_WORK_SIZE:
      msg = "if global_work_size is NULL, or if any of the values specified in global_work_size[0], ...global_work_size [work_dim - 1] are 0 or exceed the range given by the sizeof(size_t) for the device on which the kernel execution will be enqueued.";
      break;
    case CL_INVALID_GLOBAL_OFFSET:
      msg = "if the value specified in global_work_size + the corresponding values in global_work_offset for any dimensions is greater than the sizeof(size_t) for the device on which the kernel execution will be enqueued.";
      break;
    // case CL_INVALID_WORK_GROUP_SIZE:
    //   msg = "if local_work_size is specified and number of work-items specified by global_work_size is not evenly divisable by size of work-group given by local_work_size or does not match the work-group size specified for kernel using the __attribute__ ((reqd_work_group_size(X, Y, Z))) qualifier in program source.";
    //   break;
    // case CL_INVALID_WORK_GROUP_SIZE:
    //   msg = "if local_work_size is specified and the total number of work-items in the work-group computed as local_work_size[0] *... local_work_size[work_dim - 1] is greater than the value specified by CL_DEVICE_MAX_WORK_GROUP_SIZE in the table of OpenCL Device Queries for clGetDeviceInfo.";
    //   break;
    // case CL_INVALID_WORK_GROUP_SIZE:
    //   msg = "if local_work_size is NULL and the __attribute__((reqd_work_group_size(X, Y, Z))) qualifier is used to declare the work-group size for kernel in the program source.";
    //   break;
    case CL_INVALID_WORK_GROUP_SIZE:
      msg = "if local_work_size is specified and number of work-items specified by global_work_size is not evenly divisable by size of work-group given by local_work_size or does not match the work-group size specified for kernel using the __attribute__ ((reqd_work_group_size(X, Y, Z))) qualifier in program source. **OR** if local_work_size is specified and the total number of work-items in the work-group computed as local_work_size[0] *... local_work_size[work_dim - 1] is greater than the value specified by CL_DEVICE_MAX_WORK_GROUP_SIZE in the table of OpenCL Device Queries for clGetDeviceInfo. **OR** if local_work_size is NULL and the __attribute__((reqd_work_group_size(X, Y, Z))) qualifier is used to declare the work-group size for kernel in the program source.";
      break;
    case CL_INVALID_WORK_ITEM_SIZE:
      msg = "if the number of work-items specified in any of local_work_size[0], ... local_work_size[work_dim - 1] is greater than the corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES[0], .... CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1].";
      break;
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      msg = "if a sub-buffer object is specified as the value for an argument that is a buffer object and the offset specified when the sub-buffer object is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.";
      break;
    case CL_INVALID_IMAGE_SIZE:
      msg = "if an image object is specified as an argument value and the image dimensions (image width, height, specified or compute row and/or slice pitch) are not supported by device associated with queue.";
      break;
    // case CL_OUT_OF_RESOURCES:
    //   msg = "if there is a failure to queue the execution instance of kernel on the command-queue because of insufficient resources needed to execute the kernel. For example, the explicitly specified local_work_size causes a failure to execute the kernel because of insufficient resources such as registers or local memory. Another example would be the number of read-only image args used in kernel exceed the CL_DEVICE_MAX_READ_IMAGE_ARGS value for device or the number of write-only image args used in kernel exceed the CL_DEVICE_MAX_WRITE_IMAGE_ARGS value for device or the number of samplers used in kernel exceed CL_DEVICE_MAX_SAMPLERS for device.";
    //   break;
    // case CL_OUT_OF_RESOURCES:
    //   msg = "if there is a failure to allocate resources required by the OpenCL implementation on the device.";
    //   break;
    case CL_OUT_OF_RESOURCES:
      msg = "if there is a failure to queue the execution instance of kernel on the command-queue because of insufficient resources needed to execute the kernel. For example, the explicitly specified local_work_size causes a failure to execute the kernel because of insufficient resources such as registers or local memory. Another example would be the number of read-only image args used in kernel exceed the CL_DEVICE_MAX_READ_IMAGE_ARGS value for device or the number of write-only image args used in kernel exceed the CL_DEVICE_MAX_WRITE_IMAGE_ARGS value for device or the number of samplers used in kernel exceed CL_DEVICE_MAX_SAMPLERS for device. **OR** if there is a failure to allocate resources required by the OpenCL implementation on the device.";
      break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      msg = "if there is a failure to allocate memory for data store associated with image or buffer objects specified as arguments to kernel.";
      break;
    case CL_INVALID_EVENT_WAIT_LIST:
      msg = "if event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_list is not NULL and num_events_in_wait_list is 0, or if event objects in event_wait_list are not valid events.";
      break;
    case CL_OUT_OF_HOST_MEMORY:
      msg = "if there is a failure to allocate resources required by the OpenCL implementation on the host.";
      break;
  }
  return msg;
}
