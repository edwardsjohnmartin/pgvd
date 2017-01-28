#include "catch.hpp"
#include "clfw.hpp"
#include <iostream>
#include <unordered_map>

using namespace std;
using namespace cl;
TEST_CASE("CLFW can fetch a list of available compute platforms") {
  cout << "Testing CLFW platform fetching" << endl;
  WHEN("The list of platforms is initialized") {
    cl_int error = CLFW::get(CLFW::Platforms);
    THEN("the initialization should be a success.")
      REQUIRE(error == CL_SUCCESS);
  }
}
TEST_CASE("CLFW can fetch a list of compute devices") {
  WHEN("a list of devices is initialized.") {
    cout << "Testing CLFW device fetching" << endl;
    cl_int error = CLFW::get(CLFW::Devices);
    THEN("the initialization should be a success.")
      REQUIRE(error == CL_SUCCESS);
  }
  WHEN("a list of devices is initialized given a specific type of device is requested") {
    cout << "Testing CLFW GPU device fetching" << endl;
    cl_int error = CLFW::get(CLFW::Devices, CL_DEVICE_TYPE_GPU);
    THEN("the initialization should be a success.")
      REQUIRE(error == CL_SUCCESS);
  }
}
TEST_CASE("CLFW can fetch the best device") {
  cout << "Testing CLFW device selection based on characteristic." << endl;
  GIVEN("a list of devices and characteristic to use") {
    REQUIRE(CLFW::get(CLFW::Devices) == CL_SUCCESS);
    int characteristic = CL_DEVICE_MAX_CLOCK_FREQUENCY;
    THEN("the selected device has the best of that characteristic out of all the devices.") {
      cl::Device x;
      REQUIRE(CLFW::getBest(x, characteristic) == CL_SUCCESS);
      bool xMoreThanY;
      for (int i = 0; i < CLFW::Devices.size(); ++i) {
        xMoreThanY = x.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() >= CLFW::Devices[i].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
        if (!xMoreThanY)
          break;
      }
      REQUIRE(xMoreThanY);
      cout << x.getInfo<CL_DEVICE_NAME>() << endl;
    }
  }
}

TEST_CASE("Using a vector of filenames, CLFW can create an OpenCL Program") {
  GIVEN("A vector of filenames") {
    cout << "Testing CLFW program building" << endl;
    vector<string> Files = {
      "./opencl/C/BigUnsigned.c",
      "./opencl/C/ParallelAlgorithms.c",
      "./opencl/C/BuildBRT.c",
      "./opencl/C/BuildOctree.c",
      "./opencl/Kernels/kernels.cl"
    };
    THEN("We can use that vector of filenames to create a vector of sources ") {
      cl::Program::Sources sources;
      REQUIRE(CLFW::get(sources, Files) == CL_SUCCESS);

      AND_THEN("we can use those sources to build an OpenCL Program.") {
        cl::Program program;

		cl::Device bestDevice;
		REQUIRE(CLFW::getBest(bestDevice) == CL_SUCCESS);

        REQUIRE(CLFW::get(CLFW::Devices) == CL_SUCCESS);
        REQUIRE(CLFW::get(CLFW::DefaultContext, bestDevice) == CL_SUCCESS);
        REQUIRE(CLFW::Build(program, sources, CLFW::DefaultContext, bestDevice) == CL_SUCCESS);
      }
    }
  }
  WHEN("We get the sources from the opencl_sources.txt file") {
    cout << "Testing CLFW program building using opencl_sources.txt" << endl;

    cl::Program::Sources sources;
    REQUIRE(CLFW::get(sources) == CL_SUCCESS);

    THEN("We can use those sources to create an OpenCL program") {
      cl::Program program;

	  cl::Device bestDevice;
	  REQUIRE(CLFW::getBest(bestDevice) == CL_SUCCESS);

      REQUIRE(CLFW::get(CLFW::Devices) == CL_SUCCESS);
      REQUIRE(CLFW::get(CLFW::DefaultContext, bestDevice) == CL_SUCCESS);
      REQUIRE(CLFW::Build(program, sources, CLFW::DefaultContext, bestDevice) == CL_SUCCESS);
    }
  }
}

TEST_CASE("After a program is built, CLFW can use it to create kernels.") {
  cout << "Testing CLFW kernel creation." << endl;
  GIVEN("An openCL Program") {
    cl::Program::Sources sources;
    cl::Program program;
    REQUIRE(CLFW::get(sources) == CL_SUCCESS);
    REQUIRE(CLFW::get(CLFW::Devices) == CL_SUCCESS);

	cl::Device bestDevice;
	REQUIRE(CLFW::getBest(bestDevice) == CL_SUCCESS);

    REQUIRE(CLFW::get(CLFW::DefaultContext, bestDevice) == CL_SUCCESS);
    REQUIRE(CLFW::Build(program, sources, CLFW::DefaultContext, bestDevice) == CL_SUCCESS);

    THEN("We can create a hashmap of kernels.") {
      unordered_map<string, cl::Kernel> Kernels;
      REQUIRE(CLFW::get(Kernels, program) == CL_SUCCESS);
      REQUIRE(Kernels.size() > 0);
    }
  }
}

TEST_CASE("CLFW can initialize itself") {
  cout << "Testing CLFW automatic initialization" << endl;
  REQUIRE(CLFW::Initialize(true) == CL_SUCCESS);
  REQUIRE(CLFW::IsNotInitialized() == false);
}

TEST_CASE("") {
  cl::Buffer mybuffer;
  bool test;
  REQUIRE(CLFW::get(mybuffer, "key", 16, test, CLFW::DefaultContext, CL_MEM_READ_ONLY) == CL_SUCCESS);
}