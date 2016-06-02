#include "catch.hpp"
#include "KernelBox_.h"

TEST_CASE("Points can be mapped to a Z-Order curve") {
  GIVEN("a fully initialized CLFW environment") {
    if (!CLFW::IsInitialized())
      REQUIRE(CLFW::Initialize() == CL_SUCCESS);

    GIVEN("a fully initialized KernelBox") {
      if (!KernelBox::IsInitialized())
        REQUIRE(KernelBox::Initialize() == CL_SUCCESS);

      GIVEN("a couple points") {
        vector<cl_int2> points;
        for (int i = 0; i < 1000000; i++) {
          points.push_back({ i, i });
        }

        GIVEN("an OpenCL buffer that can hold those points.") {
          using namespace KernelBox;
          int globalSize = nextPow2(points.size());

          if (!isBufferUsable(buffers.points, globalSize*sizeof(cl_int2)))
            REQUIRE(createBuffer(buffers.points, globalSize*sizeof(cl_int2)) == CL_SUCCESS);
          
          GIVEN("those points are uploaded to the GPU") {
            void* gpuPoints;
            REQUIRE(buffers.points->map_buffer(gpuPoints) == CL_SUCCESS);
            memcpy(gpuPoints, points.data(), globalSize*sizeof(cl_int2));
            REQUIRE(buffers.points->unmap_buffer(gpuPoints) == CL_SUCCESS);

            THEN("the uploaded points can be mapped to a z-order curve.") {
              FAIL("Not implemented.");
              //REQUIRE(PointsToMorton(globalSize, 48) == CL_SUCCESS);

              AND_THEN("we get no race conditions.")
                FAIL("Not implemented.");
            }
          }
        }
      }
    }
  }
}