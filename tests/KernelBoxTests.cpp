#include "catch.hpp"
#include "clfw.hpp"
#include "KernelBox_.h"
#include "KernelBox_.cpp"

TEST_CASE("Using a list of files and CLFW, we can create an OpenCL program.") {
  GIVEN("A fully initialized CLFW environment") {
    if (!CLFW::IsInitialized())
      REQUIRE(CLFW::Initialize() == CL_SUCCESS);

    GIVEN("a list of source files") {
      REQUIRE(KernelBox::Files.size() > 0);
      THEN("building the OpenCL program using those files should be a success.") {
        REQUIRE(KernelBox::BuildOpenCLProgram(KernelBox::Files) == CL_SUCCESS);
      }
    }
  }
}

TEST_CASE("Using an OpenCL program, we can create kernels."){
  GIVEN("A fully initialized CLFW environment") {
    if (!CLFW::IsInitialized())
      REQUIRE(CLFW::Initialize() == CL_SUCCESS);

    GIVEN("a built OpenCL program") {
      if (!KernelBox::IsProgramInitialized())
        REQUIRE(KernelBox::BuildOpenCLProgram(KernelBox::Files) == CL_SUCCESS);

      THEN("creating the kernels should be a success.") {
        REQUIRE(KernelBox::CreateKernels(KernelBox::program) == CL_SUCCESS);
      }
    }
  }
}