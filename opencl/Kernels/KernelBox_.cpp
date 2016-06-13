#include "KernelBox_.h"

namespace KernelBox {
  bool initialized = false;
  cl::Program program;
  cl::Program::Sources sources;

  int nextPow2(int num) { return max((int)pow(2, ceil(log(num) / log(2))), 8); }
  
  //Buffers buffers;

//  bool isBufferUsable(shared_ptr<Buffer> buffer, size_t expectedSizeInBytes) {
//    if (buffer == nullptr)
//      return false;
//    else if (buffer->getSize() < expectedSizeInBytes)
//      return false;
//    else
//      return true;
//  }
//  cl_int createBuffer(shared_ptr<Buffer> &buffer, size_t size) {
//    if (size <= 0) return CL_INVALID_BUFFER_SIZE;
//    buffer = make_shared<Buffer>(size, CLFW::DefaultContext, CLFW::Queues[0]);
//    if (!buffer) return CL_INVALID_PROPERTY;
//    return CL_SUCCESS;
//  }
//
//  cl_int Initialize() {
//    if (!initialized) {
//      cl_int error = 0;
//      error |= BuildOpenCLProgram(Files);
//     // error |= CreateKernels(program);
//      if (error == CL_SUCCESS)
//        initialized = true;
//      else {
//        initialized == false;
//        return error;
//      }
//    }
//    return CL_SUCCESS;
//  }
//
}