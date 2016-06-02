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
#include <CL/opencl.h>
#endif
#include <vector>
#include <string>

using namespace std;
static class CLFW{

public:
  /* Flags */
  static bool verbose;
  
  /* Member Variables */
  static cl_context Context;

  /* Member Lists */
  static vector<cl_platform_id> Platforms;
  static vector<cl_device_id> Devices;
  static vector<cl_command_queue> Queues;

  /* Queries */
  static string GetPlatformName(cl_platform_id id);
  static string GetDeviceName(cl_device_id id);
  static bool IsInitialized();

  /* Initializers */
  static cl_int Initialize(bool _verbose = false);
  static cl_int InitializePlatformList();
  static cl_int InitializeDeviceList(int platformIndex, int deviceType = CL_DEVICE_TYPE_ALL);
  static cl_int InitializeContext(int platformIndex = 0);

  static cl_int AddQueue(int deviceIndex);

  /* Terminators */
  static cl_int Terminate();
};