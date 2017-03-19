#include <iostream>
#include "clfw.hpp"

/* Verbose things */
bool CLFW::verbose = false;
cl_int CLFW::error = false;
bool CLFW::lastBufferOld = false;

void CLFW::print(std::string s, bool forced) {
	if (verbose || error || forced)
		printf("CLFW: %-70s\n", s.c_str());
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
cl::DeviceCommandQueue CLFW::DeviceQueue;

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

/* Initializers */
cl_int CLFW::Initialize(bool _verbose, int chosenDevice, std::string buildOptions, unsigned int numQueues) {
	verbose = _verbose;
	print("Initializing...");
	error = getPlatforms(Platforms);
	error |= getDevices(Devices);

	if (chosenDevice == -1 || chosenDevice > Devices.size())
		error |= queryDevice(DefaultDevice);
	else
		DefaultDevice = Devices[chosenDevice];

	error |= getContext(DefaultContext);
	Contexts.push_back(DefaultContext);

	for (int i = 0; i < numQueues; ++i) {
		cl::CommandQueue queue;
		error |= getQueue(queue);
		Queues.push_back(queue);
		if (i == 0)
			DefaultQueue = queue;
		if (i == 1)
			SecondaryQueue = queue;
	}
	cl_uint maxQueueSize = DefaultDevice.getInfo<CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE>();
	DeviceQueue = cl::DeviceCommandQueue(DefaultContext, DefaultDevice, maxQueueSize, cl::DeviceQueueProperties::None, &error);

	error |= getSources(DefaultSources);
	error |= buildProgram(DefaultProgram, DefaultSources, DefaultContext, DefaultDevice, buildOptions);
	error |= getKernels(Kernels);
	return error;
}

/* Accessors */
cl_int CLFW::getPlatforms(std::vector<cl::Platform> &Platforms) {
	error = cl::Platform::get(&Platforms);
	if (Platforms.size() == 0) {
		print("No platforms found. Check OpenCL installation!");
		return error;
	}

	print("Found " + std::to_string(Platforms.size()) + " platforms(s)");
	for (cl_uint i = 0; i < Platforms.size(); ++i)
		print("[" + std::to_string(i) + "] -> " + Platforms[i].getInfo<CL_PLATFORM_NAME>());
  
	return CL_SUCCESS;
}
cl_int CLFW::getDevices(std::vector<cl::Device> &Devices, int deviceType) {
	if (Platforms.size() == 0) {
		error = CL_INVALID_VALUE;
		print("No platforms found. Check OpenCL installation!");
		return error;
	}
  
	Devices.resize(0);
  
	error = 0;
	std::vector<cl::Device> temp;
	for (int i = 0; i < Platforms.size(); ++i) {
		temp.clear();
		error |= Platforms[i].getDevices(deviceType, &temp);
		Devices.insert(Devices.end(), temp.begin(), temp.end());
	}

	if (Devices.size() == 0) {
		error = CL_INVALID_DEVICE;
		print("No devices found. Check OpenCL installation!");
		return error;
	}

	print("Found " + std::to_string(Devices.size()) + " device(s) for " + std::to_string(Platforms.size()) + " platform(s)");
	for (cl_uint i = 0; i < Devices.size(); ++i)
		print("[" + std::to_string(i) + "] : " + Devices[i].getInfo<CL_DEVICE_NAME>());

	return error;
}
cl_int CLFW::getContext(cl::Context &context, const cl::Device &device) {
	error = 0;
	context = cl::Context({ device }, NULL, NULL, NULL, &error);
	if (error == CL_SUCCESS) print("Created context for " + device.getInfo<CL_DEVICE_NAME>());
	else print("Failed creating context for " + device.getInfo<CL_DEVICE_NAME>());
	return error;
}
cl_int CLFW::getQueue(cl::CommandQueue &queue, const cl::Context &context, const cl::Device &device) {
	error = 0;
	queue = cl::CommandQueue(context, device, error);
	if (error == CL_SUCCESS) print("Created queue for DefaultContext's " + device.getInfo<CL_DEVICE_NAME>());
	else print("Failed creating queue for DefaultContext's " + device.getInfo<CL_DEVICE_NAME>());
	return error;
}
cl_int CLFW::getSources(cl::Program::Sources &sources, std::vector<std::string> &files) {
	sources.clear();
	error = 0;
	if (files.size() == 0) return CL_INVALID_VALUE;
	for (int i = 0; i < files.size(); ++i) {
		print("adding " + files[i] + " to sources.");
		long length = 0;
		char *source = 0;
		error|=loadFile(files[i].c_str(), &source, &length);
		sources.push_back({ source, (unsigned __int64)length });
	}
	return error;
}
cl_int CLFW::getSources(cl::Program::Sources &sources) {
	sources.clear();
	char* text = 0;
	long temp;
	loadFile(OPENCL_SOURCES_PATH, &text, &temp);
	try {
		char* file = strtok(text, "\n\r\n\0");
		do {
			print("adding " + std::string(file) + " to sources.");
			long length = 0;
			char *source = 0;
			loadFile(file, &source, &length);
			sources.push_back({ source, (unsigned __int64)length });
			file = strtok(NULL, "\n\r\n\0");
		} while (file != NULL);
	}
	catch (...) {
		error |= CL_INVALID_ARG_VALUE;
		print("Unable to open ./Sources/OpenCL/opencl_sources.txt!!");
	}
	return error;
}
cl_int CLFW::getKernels(std::unordered_map<cl::string, cl::Kernel> &Kernels, cl::Program &program) {
	Kernels.clear();
	std::vector<cl::Kernel> tempKernels;
	error = program.createKernels(&tempKernels);
	if (error != CL_SUCCESS) {
		print("Unable to create kernels.");
		return error;
	}
	for (int i = 0; i < tempKernels.size(); ++i) {
		std::string temp = std::string(tempKernels[i].getInfo<CL_KERNEL_FUNCTION_NAME>());
		Kernels[temp] = tempKernels[i];
	}
	for (auto i : Kernels) {
		print("Created Kernel " + i.first);
	}
	return CL_SUCCESS;
}
cl_int CLFW::getBuffer(cl::Buffer &buffer, std::string key, cl_ulong size, bool &old, cl::Context &context, int flag) {
	error = 0;
	old = true;
	//If the key is not found...
	if (Buffers.find(key) == Buffers.end()) old = false;
	else if (Buffers[key].getInfo<CL_MEM_SIZE>() != size) old = false;

	if (old == false) 
	{
		Buffers[key] = cl::Buffer(context, flag, size, NULL, &error);
		if (error == CL_SUCCESS)
			print("Created buffer " + key + " of size " + std::to_string(size) + " bytes");
		else
			print("Failed to create buffer " + key + " of size " + std::to_string(size) + " bytes. Error #  %d");
	}
	buffer = Buffers[key];
	return error;
}
cl_int CLFW::getBestDevice(cl::Device &device, int characteristic) {
	int largest = 0;
	int temp;

	const int COMBINED = CL_DEVICE_MAX_CLOCK_FREQUENCY & CL_DEVICE_MAX_COMPUTE_UNITS & CL_DEVICE_MAX_WORK_GROUP_SIZE;
  
	if (characteristic != CL_DEVICE_MAX_CLOCK_FREQUENCY &&
			characteristic != CL_DEVICE_MAX_COMPUTE_UNITS &&
		characteristic != CL_DEVICE_MAX_WORK_GROUP_SIZE &&
		characteristic != COMBINED)
	{
		error |= CL_INVALID_ARG_VALUE;
		print("Device characteristic unrecognized! ");
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
	print("Selected " + device.getInfo<CL_DEVICE_NAME>());
	return CL_SUCCESS;
}

cl_int CLFW::queryDevice(cl::Device &device) {
	if (Devices.size() == 0) return CL_INVALID_ARG_SIZE;
	
	print("Which device would you like to use? (enter a number between 0 and " + std::to_string(Devices.size()-1) + ")", true);

	for (cl_uint i = 0; i < Devices.size(); ++i) {
		print("[" + std::to_string(i) + "] : " + Devices[i].getInfo<CL_DEVICE_NAME>() 
			+ " " + Devices[i].getInfo<CL_DEVICE_VERSION>().c_str(), true);
		print(Devices[i].getInfo<CL_DEVICE_VERSION>(), true);
	}


	int selection;
	do {
		while (!(std::cin >> selection)) {
			std::cin.clear();
			while (std::cin.get() != '\n') continue;
		}
	} while (selection >= Devices.size());

	device = Devices[selection];
	print("Selected " + device.getInfo<CL_DEVICE_NAME>());
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

cl_int CLFW::buildProgram(cl::Program &program, cl::Program::Sources &sources, cl::Context &context, cl::Device &device, std::string options) {
	//char* spir;
	//long length;
	//cl_int spirError;
	//loadFile("./kernel.spir", &spir, (long*)&length);
	//cl_program spirProgram = clCreateProgramWithIL(context(), spir, length, &error);

	//program = cl::Program(spirProgram);
	program = cl::Program(context, sources, &error);


	if (error != CL_SUCCESS) {
		print("Error creating program:");
		return error;
	}

	std::cout << "Building cl program with options \"" << options << "\"" << std::endl;
	error = program.build({ device }, options.c_str());
	if (error != CL_SUCCESS) {
		print("Error building program:");
		print(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
	}
	else {
		print("Success building OpenCL program. ");
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
