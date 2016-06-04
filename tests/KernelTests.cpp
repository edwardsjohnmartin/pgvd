#include "catch.hpp"
#include "clfw.hpp"
#include "Kernels.h"

#define OneMillion 1000000
#define NumLevels 30

SCENARIO("Points can be mapped to a Z-Order curve") {
  GIVEN("a big unsigned can hold " + to_string(NumLevels) + " Z-Order levels") {
    REQUIRE(NumLevels * DIM <= 8 * BIG_INTEGER_SIZE);

    GIVEN("a fully initialized CLFW environment") {
      if (!CLFW::IsInitialized()) REQUIRE(CLFW::Initialize() == CL_SUCCESS);

      GIVEN("a fully initialized KernelBox") {
        if (!KernelBox::IsInitialized()) REQUIRE(KernelBox::Initialize() == CL_SUCCESS);

        GIVEN("a couple points") {
          vector<cl_int2> points;
          for (int i = 0; i < OneMillion; i++) points.push_back({ i, i });
          
          GIVEN("an OpenCL buffer that can hold those points.") {
            using namespace KernelBox;
            int globalSize = nextPow2(points.size());

            //Create a points buffer, but only if it hasn't been created in another test.
            if (!isBufferUsable(buffers.points, globalSize*sizeof(cl_int2)))
              REQUIRE(createBuffer(buffers.points, globalSize*sizeof(cl_int2)) == CL_SUCCESS);

            GIVEN("those points are uploaded to the GPU") {
              void* gpuPoints;
              REQUIRE(buffers.points->map_buffer(gpuPoints) == CL_SUCCESS);
              memcpy(gpuPoints, points.data(), globalSize*sizeof(cl_int2));
              REQUIRE(buffers.points->unmap_buffer(gpuPoints) == CL_SUCCESS);

              THEN("the uploaded points can be mapped to a z-order curve.") {
                //Call the parallel kernel
                REQUIRE(PointsToMorton_p(points.size(), NumLevels) == CL_SUCCESS);

                AND_THEN("we get no race conditions.") {
                  vector<BigUnsigned> result(globalSize);
                  void* data;
                  BigUnsigned* zPoints;
                  REQUIRE(PointsToMorton_s(points.size(), NumLevels, points.data(), result.data()) == 0);
                  REQUIRE(buffers.bigUnsignedInput->map_buffer(data) == CL_SUCCESS);
                  zPoints = (BigUnsigned*) data;

                  //Compare the serial calculations with the parallel calculations.
                  int compareResult = 0;
                  for (int i = 0; i < globalSize; ++i) {
                    compareResult = compareBU(&zPoints[i], &result[i]);
                    if (compareResult != 0)
                      break;
                  }
                  REQUIRE(compareResult == 0);
                  
                  //Free the mapped buffer pointer.
                  REQUIRE(buffers.bigUnsignedInput->unmap_buffer(data) == CL_SUCCESS);
                }
              }
            }
          }
        }
      }
    }
  }
}