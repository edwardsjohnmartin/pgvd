#include "catch.hpp"
#include "clfw.hpp"
#include "Kernels.h"

#define OneMillion 1000000
#define NumLevels 30

//This should really be in a CPP file...
inline std::string buToString(BigUnsigned bu) {
  std::string representation = "";
  if (bu.len == 0)
  {
    representation += "[0]";
  }
  else {
    for (int i = bu.len; i > 0; --i) {
      representation += "[" + std::to_string(bu.blk[i - 1]) + "]";
    }
  }

  return representation;
}

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

SCENARIO("Big Unsigneds can be sorted using a parallel radix sort.") {
  GIVEN("a fully initialized CLFW environment") {
    if (!CLFW::IsInitialized()) REQUIRE(CLFW::Initialize() == CL_SUCCESS);

    GIVEN("a fully initialized KernelBox") {
      if (!KernelBox::IsInitialized()) REQUIRE(KernelBox::Initialize() == CL_SUCCESS);

      GIVEN("a couple BigUnsigned numbers.") {
        using namespace KernelBox;
        vector<BigUnsigned> hostNumbers(nextPow2(OneMillion));
        for (int i = 0; i < OneMillion; ++i) {
          int temp = i%(8*BIG_INTEGER_SIZE);
          initBlkBU(&hostNumbers[i], 1);
          shiftBULeft(&hostNumbers[i], &hostNumbers[i], temp);
        }
        for (int i = OneMillion; i < hostNumbers.size(); ++i) {
          initBlkBU(&hostNumbers[i], 0);
        }
        GIVEN("an OpenCL buffer that can hold those numbers.") {
          int globalSize = nextPow2(hostNumbers.size());

          //Create a BU buffer, but only if it hasn't been created in another test.
          if (!isBufferUsable(buffers.bigUnsignedInput, globalSize*sizeof(BigUnsigned)))
            REQUIRE(createBuffer(buffers.bigUnsignedInput, globalSize*sizeof(BigUnsigned)) == CL_SUCCESS);
          
          GIVEN("those numbers are uploaded sucessfully to the GPU") {
            void* data;
            REQUIRE(buffers.bigUnsignedInput->map_buffer(data) == CL_SUCCESS);
            memcpy(data, hostNumbers.data(), hostNumbers.size()*sizeof(BigUnsigned));
            REQUIRE(buffers.bigUnsignedInput->unmap_buffer(data) == CL_SUCCESS);

            THEN("we can sort those numbers with a parallel radix sort routine.") {
              REQUIRE(RadixSortBigUnsigned(hostNumbers.size(), BIG_INTEGER_SIZE*8) == CL_SUCCESS);

              AND_THEN("There are no race conditions.") {
                std::sort(hostNumbers.begin(), hostNumbers.end(), weakCompareBU);

                REQUIRE(buffers.bigUnsignedInput->map_buffer(data) == CL_SUCCESS);
                BigUnsigned* GPUNumbers = (BigUnsigned*)data;

                //Compare the serial calculations with the parallel calculations.
                int compareResult = 0;
                for (int i = 0; i < globalSize; ++i) {
                  compareResult = compareBU(&hostNumbers[i], &GPUNumbers[i]);
                  if (compareResult != 0)
                    break;
                }
                REQUIRE(compareResult == 0);

                REQUIRE(buffers.bigUnsignedInput->unmap_buffer(data) == CL_SUCCESS);
              }
            }
          }
        }
      }
    }
  }
}

SCENARIO("Sorted Z-Order numbers can be used to construct a binary radix tree") {
  GIVEN("a fully initialized CLFW environment") {
    if (!CLFW::IsInitialized()) REQUIRE(CLFW::Initialize() == CL_SUCCESS);

    GIVEN("a fully initialized KernelBox") {
      if (!KernelBox::IsInitialized()) REQUIRE(KernelBox::Initialize() == CL_SUCCESS);

      /* initialize random seed: */
      srand(time(NULL));

      GIVEN("a couple sorted, uniqued points") {
        vector<BigUnsigned> zpoints(OneMillion);
        for (int i = 0; i < OneMillion; ++i) {
          initBU(&zpoints[i]);
          for (int j = 0; j < BIG_INTEGER_SIZE; j++) {
            zpoints[i].blk[j] = rand();
          }
        }

        sort(zpoints.begin(), zpoints.end(), weakCompareBU);
        auto last = unique(zpoints.begin(), zpoints.end(), weakEqualsBU);
        zpoints.erase(last, zpoints.end());

        GIVEN("an OpenCL buffer that can hold those numbers.") {
          using namespace KernelBox;
          int globalSize = nextPow2(zpoints.size());

          //Create a BU buffer, but only if it hasn't been created in another test.
          if (!isBufferUsable(buffers.bigUnsignedInput, globalSize*sizeof(BigUnsigned)))
            REQUIRE(createBuffer(buffers.bigUnsignedInput, globalSize*sizeof(BigUnsigned)) == CL_SUCCESS);

          GIVEN("those points are then uploaded to the GPU") {
            void* data;
            REQUIRE(buffers.bigUnsignedInput->map_buffer(data) == CL_SUCCESS);
            memcpy(data, zpoints.data(), zpoints.size() * sizeof(BigUnsigned));
            REQUIRE(buffers.bigUnsignedInput->unmap_buffer(data) == CL_SUCCESS);

            THEN("we can build a BRT with those sorted, uniqued points in parallel") {
              REQUIRE(BuildBinaryRadixTree_p(zpoints.size(), BIG_INTEGER_SIZE * 8) == CL_SUCCESS);

              AND_THEN("we get no race conditions.") {
                //Determine what the BRT should look like
                vector<BrtNode> I(zpoints.size() - 1);
                vector<BrtNode> L(zpoints.size());
                REQUIRE(BuildBinaryRadixTree_s(zpoints.size(), BIG_INTEGER_SIZE * 8, zpoints.data(), I.data(), L.data()) == CL_SUCCESS);

                //Fetch what the BRT actually looks like.
                vector<BrtNode> GPU_I(zpoints.size() - 1);
                REQUIRE(buffers.internalNodes->map_buffer(data) == CL_SUCCESS);
                memcpy(GPU_I.data(), data, GPU_I.size()*sizeof(BrtNode));
                REQUIRE(buffers.internalNodes->unmap_buffer(data) == CL_SUCCESS);

                bool compareResult = true;
                for (int i = 0; i < I.size(); ++i) {
                  compareResult = compareBrtNode(&GPU_I[i], &I[i]);
                  if (compareResult == false) {
                    break;
                  }
                }
                REQUIRE(compareResult == true);
              }
            }
          }
        }
      }
    }
  }
}