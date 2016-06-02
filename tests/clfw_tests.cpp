#include "catch.hpp"
#include "clfw.hpp"

TEST_CASE("CLFW can fetch a list of available compute platforms") {
  WHEN("The list of platforms is initialized") {
    cl_int error = CLFW::InitializePlatformList();
    THEN("the initialization should be a success.")
      REQUIRE(error == CL_SUCCESS);
    AND_THEN( "at least one platform should be in the list" )
      REQUIRE(CLFW::Platforms.size() > 0);
  }
}

TEST_CASE("CLFW can fetch a list of compute devices for a given platform") {
  GIVEN("An initialized list of platforms") {
    REQUIRE(CLFW::InitializePlatformList() == CL_SUCCESS);
    WHEN("a list of devices is initialized for each platform.") {
      for (int i = 0; i < CLFW::Platforms.size(); ++i) {
        cl_int error = CLFW::InitializeDeviceList(i);
        THEN("the initialization for each platform should be a success.")
          REQUIRE(error == CL_SUCCESS);
        AND_THEN("at least one device should be in each list. ")
          REQUIRE(CLFW::Devices.size() > 0);
      }
    }
    THEN("a list of a specific type of devices can be initialized for each platform.") {
      for (int i = 0; i < CLFW::Platforms.size(); ++i) {
        cl_int error = CLFW::InitializeDeviceList(i, CL_DEVICE_TYPE_GPU);
        THEN("the initialization for each platform should be a success.")
          REQUIRE(error == CL_SUCCESS);
      }
    }
  }
}

TEST_CASE("CLFW can create an OpenCL context.") {
  GIVEN("A list of devices") {
    REQUIRE(CLFW::InitializePlatformList() == CL_SUCCESS);
    REQUIRE(CLFW::Platforms.size() > 0);
    REQUIRE(CLFW::InitializeDeviceList(0) == CL_SUCCESS);
    REQUIRE(CLFW::Devices.size() > 0);
    WHEN("CLFW initializes a context") {
      cl_int error = CLFW::InitializeContext(0);
      THEN("the initialization should be a success") {
        REQUIRE(error == CL_SUCCESS);
      }
    }
  }
}

TEST_CASE("CLFW can add and remove an OpenCL queue.") {
  GIVEN("an OpenCL Context and a list of devices.") {
    REQUIRE(CLFW::InitializePlatformList() == CL_SUCCESS);
    REQUIRE(CLFW::Platforms.size() > 0);
    REQUIRE(CLFW::InitializeDeviceList(0) == CL_SUCCESS);
    REQUIRE(CLFW::Devices.size() > 0);
    REQUIRE(CLFW::InitializeContext(0) == CL_SUCCESS);

    WHEN("we add a queue for a particular device") {
      cl_int error = CLFW::AddQueue(0);
      THEN("AddQueue should be successful.")
        REQUIRE(error == CL_SUCCESS);
    }
  }
}