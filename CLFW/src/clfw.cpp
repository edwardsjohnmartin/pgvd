#include <iostream>
#include "clfw.hpp"

using namespace std;
using namespace cl;
/* Verbose things */
bool CLFW::verbose = true;
bool CLFW::lastBufferOld = false;
void CLFW::Print(string s, int fgcode, int bgcode) {
  if (verbose || ((fgcode == errorFG) && (bgcode = errorBG))) 
    printf("\033[%d;%dmCLFW: %-74s\033[0m\n",fgcode, bgcode, s.c_str());
}

/* Source file management */
void CLFW::loadFile(const char* name, char** buffer, long* length)
{
  FILE * f = fopen(name, "rb");
  if (f)
  {
    fseek(f, 0, SEEK_END);
    *length = ftell(f);
    fseek(f, 0, SEEK_SET);
    *buffer = (char*)malloc(*length + 1);
    (*buffer)[*length] = 0;
    if (*buffer)
    {
      fread(*buffer, 1, *length, f);
    }
    fclose(f);
  }
}

/* Member Variables */
Device CLFW::DefaultDevice;
Context CLFW::DefaultContext;
CommandQueue CLFW::DefaultQueue;

Program CLFW::DefaultProgram;
Program::Sources CLFW::DefaultSources;

/* Lists */
vector<Platform> CLFW::Platforms;
vector<Device> CLFW::Devices;
vector<Context> CLFW::Contexts;
vector<CommandQueue> CLFW::Queues;

/* Maps */
unordered_map<string, Kernel> CLFW::Kernels;
unordered_map<string, Buffer> CLFW::Buffers;

/* Queries */
bool CLFW::IsNotInitialized() {
  if (Platforms.size() == 0) return true;
  if (Devices.size() == 0) return true;
  if (Contexts.size() == 0) return true;
  if (Queues.size() == 0) return true;
  return false;
}

/* Initializers */
cl_int CLFW::Initialize(bool _verbose, int characteristic) {
  verbose = _verbose;
  if (verbose) Print("Initializing...", infoFG, infoBG);
  cl_int  error = get(Platforms);
  error |= get(Devices);
  error |= getBest(DefaultDevice, characteristic);
  Contexts.clear();
  error |= get(DefaultContext);
  Contexts.push_back(DefaultContext);
  Queues.clear();
  error |= get(DefaultQueue);
  Queues.push_back(DefaultQueue);
  error |= get(DefaultSources);
  error |= Build(DefaultProgram, DefaultSources);
  error |= get(Kernels);
  
  return error;
}

/* Accessors */
cl_int CLFW::get(vector<Platform> &Platforms) {
  cl_int error = cl::Platform::get(&Platforms);
  if (Platforms.size() == 0) {
    Print("No platforms found. Check OpenCL installation!", errorFG, errorBG);
    return error;
  }

  Print("Found " + to_string(Platforms.size()) + " platforms(s)", successFG, successBG);
  for (cl_uint i = 0; i < Platforms.size(); ++i)
    Print("[" + to_string(i) + "] -> " + Platforms[i].getInfo<CL_PLATFORM_NAME>(), infoFG, infoBG);
  
  return CL_SUCCESS;
}
cl_int CLFW::get(vector<Device> &Devices, int deviceType) {
  if (Platforms.size() == 0) {
    Print("No platforms found. Check OpenCL installation!", errorFG, errorBG);
    return CL_INVALID_VALUE;
  }
  
  Devices.resize(0);
  
  cl_int error = 0;
  vector<Device> temp;
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
    Print("Found " + to_string(Devices.size()) + " device(s) for " + to_string(Platforms.size()) + " platform(s)", successFG, successBG);
  for (cl_uint i = 0; i < Devices.size(); ++i)
    Print("[" + to_string(i) + "] : " + Devices[i].getInfo<CL_DEVICE_NAME>(), infoFG, infoBG);

  return CL_SUCCESS;
}
cl_int CLFW::get(Context &context, const Device &device) {
  cl_int error = 0;
  context = Context({ device }, NULL, NULL, NULL, &error);
  if (error == CL_SUCCESS) Print("Created context for " + device.getInfo<CL_DEVICE_NAME>(), successFG, successBG);
  else Print("Failed creating context for " + device.getInfo<CL_DEVICE_NAME>(), errorFG, errorBG);
  return error;
}
cl_int CLFW::get(CommandQueue &queue, const Context &context, const Device &device) {
  cl_int error = 0;
  queue = CommandQueue(context, device, error);
  if (error == CL_SUCCESS) Print("Created queue for DefaultContext's " + device.getInfo<CL_DEVICE_NAME>(), successFG, successBG);
  else Print("Failed creating queue for DefaultContext's " + device.getInfo<CL_DEVICE_NAME>(), errorFG, errorBG);
  return error;
}
cl_int CLFW::get(Program::Sources &sources, vector<string> &files) {
  sources.clear();
  if (files.size() == 0) return CL_INVALID_VALUE;
  for (int i = 0; i < files.size(); ++i) {
    Print("adding " + files[i] + " to sources.", infoFG, infoBG);
    long length = 0;
    char *source = 0;
    loadFile(files[i].c_str(), &source, &length);
    sources.push_back({ source, length });
  }
  return CL_SUCCESS;
}
cl_int CLFW::get(Program::Sources &sources) {
  sources.clear();
  char* text = 0;
  long temp;
  loadFile("./opencl_sources.txt", &text, &temp);
  char* file = strtok(text, "\n\r\n\0");
  do {
    Print("adding " + string(file) + " to sources.", infoFG, infoBG);
    long length = 0;
    char *source = 0;
    loadFile(file, &source, &length);
    sources.push_back({ source, length });
    file = strtok(NULL, "\n\r\n\0");
  } while (file != NULL);
  return CL_SUCCESS;
}
cl_int CLFW::get(unordered_map<STRING_CLASS, cl::Kernel> &Kernels, cl::Program &program) {
  Kernels.clear();
  vector<Kernel> tempKernels;
  cl_int error = program.createKernels(&tempKernels);
  if (error != CL_SUCCESS) {
    Print("Unable to create kernels.", errorFG, errorBG);
    return error;
  }
  for (int i = 0; i < tempKernels.size(); ++i) {
    string temp = string(tempKernels[i].getInfo<CL_KERNEL_FUNCTION_NAME>());
    
    //For some reason, OpenCL string's lengths are 1 char longer than they should be.
    temp = temp.substr(0, temp.length() - 1);
    Kernels[temp] = tempKernels[i];
  }
  for (auto i : Kernels) {
    Print("Created Kernel " + i.first, successFG, successBG);
  }
  return CL_SUCCESS;
}
cl_int CLFW::get(cl::Buffer &buffer, std::string key, std::size_t size, bool &old, cl::Context &context, int flag) {
  if (Buffers[key].getInfo<CL_MEM_SIZE>() != size) {
    Buffers[key] = Buffer(context, flag, size);
    Print("Created buffer " + key + " of size " + to_string(size) + " bytes" , successFG, successBG);
    old = false;
  } else old = true;
  buffer = Buffers[key];
  return CL_SUCCESS;
}

cl_int CLFW::getBest(Device &device, int characteristic) {
  cl_int error;
  int largest = 0;
  int temp;

  if (characteristic != CL_DEVICE_MAX_CLOCK_FREQUENCY &&
      characteristic != CL_DEVICE_MAX_COMPUTE_UNITS)
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
    }
    if (largest < temp) {
      largest = temp;
      device = Devices[i];
    }
  }
  Print("Selected " + device.getInfo<CL_DEVICE_NAME>(), successFG, successBG);
  return CL_SUCCESS;
}

cl_int CLFW::Build(cl::Program &program, cl::Program::Sources &sources, cl::Context &context, cl::Device &device) {
  cl_int error;
  program = cl::Program(context, sources);
  error = program.build({ device });
  if (error != CL_SUCCESS) {
    Print("Error building program:", errorFG, errorBG);
    Print(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device), errorFG, errorBG);
  }
  else {
    Print("Success building OpenCL program. ", successFG, successBG);
  }
  
  return error;
}