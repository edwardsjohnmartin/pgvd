#include "catch.hpp"
#include "clfw.hpp"
#include "Kernels.h"
#include <iostream>

#define OneMillion 1000000
#define bits 30
#define mbits bits*DIM

static bool brtTestPassed;
static unsigned int zpointsSize;
static cl::Buffer internalBRTNodes;

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

SCENARIO("Points can be uploaded to the GPU.") {
  cout << "Testing PointsToMorton kernel" << endl;
  GIVEN("a fully initialized CLFW environment") {
    if (CLFW::IsNotInitialized()) REQUIRE(CLFW::Initialize() == CL_SUCCESS);
    GIVEN("a couple points") {
      vector<cl_int2> points;
      for (int i = 0; i < OneMillion; i++) points.push_back({ i, i });

      GIVEN("a buffer that can hold some points") {
        cl::Buffer buffer(CLFW::DefaultContext, CL_MEM_READ_WRITE, sizeof(cl_int2)*points.size());
        REQUIRE(CLFW::DefaultQueue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(cl_int2) * points.size(), points.data()) == CL_SUCCESS);
      }
    }
  }
}

SCENARIO("Points can be mapped to a Z-Order curve") {
  cout << "Testing PointsToMorton kernel" << endl;
  GIVEN("a big unsigned can hold " + to_string(bits) + " Z-Order levels") {
    REQUIRE(mbits <= 8 * BIG_INTEGER_SIZE);

    GIVEN("a fully initialized CLFW environment") {
      if (CLFW::IsNotInitialized()) REQUIRE(CLFW::Initialize() == CL_SUCCESS);

      GIVEN("a couple points") {
        vector<cl_int2> points;
        for (int i = 0; i < OneMillion; i++) points.push_back({ i, i });
          
        GIVEN("an OpenCL buffer that can hold those points.") {
          using namespace Kernels;
          int globalSize = nextPow2(points.size());
          cl::Buffer pointsBuffer;
          REQUIRE(CLFW::get(pointsBuffer, "points", globalSize*sizeof(cl_int2)) == CL_SUCCESS);

          WHEN("those points are uploaded to the GPU") {
            REQUIRE(CLFW::DefaultQueue.enqueueWriteBuffer(pointsBuffer, CL_TRUE, 0, points.size() * sizeof(cl_int2), points.data()) == CL_SUCCESS);
            THEN("the uploaded points can be mapped to a z-order curve.") {
              cl::Buffer zPointsBuffer;
              REQUIRE(PointsToMorton_p(pointsBuffer, zPointsBuffer, points.size(), bits) == CL_SUCCESS);

              AND_THEN("we get no race conditions.") {
                vector<BigUnsigned> hostZPoints(globalSize);
                vector<BigUnsigned> GPUZPoints(globalSize);
                REQUIRE(CLFW::DefaultQueue.enqueueReadBuffer(zPointsBuffer, CL_TRUE, 0, globalSize*sizeof(BigUnsigned), GPUZPoints.data()) == CL_SUCCESS);
                REQUIRE(PointsToMorton_s(points.size(), bits, points.data(), hostZPoints.data()) == CL_SUCCESS);

                //Compare the serial calculations with the parallel calculations.
                int compareResult = 0;
                for (int i = 0; i < globalSize; ++i) {
                  compareResult = compareBU(&hostZPoints[i], &GPUZPoints[i]);
                  if (compareResult != 0)
                    break;
                }
                REQUIRE(compareResult == 0);
              }
            }
          }
        }
      }
    }
  }
}

SCENARIO("Big Unsigneds can be sorted using a parallel radix sort.") {
  cout << "Testing parallel radix sort" << endl;
  GIVEN("a fully initialized CLFW environment") {
    if (CLFW::IsNotInitialized()) REQUIRE(CLFW::Initialize() == CL_SUCCESS);
    /* initialize random seed: */
    srand(time(NULL));
    GIVEN("a couple BigUnsigned numbers.") {
      using namespace Kernels;
      vector<BigUnsigned> hostNumbers(nextPow2(OneMillion));
      for (int i = 0; i < OneMillion; ++i) {
        initBU(&hostNumbers[i]);
        for (int j = 0; j < (mbits); j++) {
          setBUBit(&hostNumbers[i], j, rand()%2);
        }
        hostNumbers[i].len = (mbits)/8;
        zapLeadingZeros(&hostNumbers[i]);
      }
      for (int i = OneMillion; i < hostNumbers.size(); ++i) {
        initBlkBU(&hostNumbers[i], 0);
      }
      GIVEN("an OpenCL buffer that can hold those numbers.") {
        int globalSize = nextPow2(hostNumbers.size());

        cl::Buffer buffer;
        REQUIRE(CLFW::get(buffer, "buffer", globalSize*sizeof(BigUnsigned)) == CL_SUCCESS);

        GIVEN("those numbers are uploaded sucessfully to the GPU") {
          REQUIRE(CLFW::DefaultQueue.enqueueWriteBuffer(buffer, CL_TRUE, 0, hostNumbers.size()*sizeof(BigUnsigned), hostNumbers.data()) == CL_SUCCESS);
      
          THEN("we can sort those numbers with a parallel radix sort routine.") {
            REQUIRE(RadixSortBigUnsigned(buffer, hostNumbers.size(), BIG_INTEGER_SIZE*8) == CL_SUCCESS);

            AND_THEN("There are no race conditions.") {
              std::sort(hostNumbers.begin(), hostNumbers.end(), weakCompareBU);

              vector<BigUnsigned> GPUNumbers(globalSize);
              REQUIRE(CLFW::DefaultQueue.enqueueReadBuffer(buffer, CL_TRUE, 0, globalSize*sizeof(BigUnsigned), GPUNumbers.data()) == CL_SUCCESS);

              //Compare the serial calculations with the parallel calculations.
              int compareResult = 0;
              for (int i = 0; i < globalSize; ++i) {
                compareResult = compareBU(&hostNumbers[i], &GPUNumbers[i]);
                if (compareResult != 0)
                  break;
              }
              REQUIRE(compareResult == 0);
            }
          }
        }
      }
    }
  }
}

SCENARIO("Sorted BigUnsigneds can be unique'd in parallel.") {
  cout << "Testing UniqueSorted kernel" << endl;
  GIVEN("a fully initialized CLFW environment") {
    if (CLFW::IsNotInitialized()) REQUIRE(CLFW::Initialize() == CL_SUCCESS);

    GIVEN("a couple BigUnsigned numbers.") {
      using namespace Kernels;
      vector<BigUnsigned> hostNumbers(nextPow2(OneMillion));
      for (int i = 0; i < OneMillion; ++i) {
        initBU(&hostNumbers[i]);
        for (int j = 0; j < (mbits); j++) {
          setBUBit(&hostNumbers[i], j, rand() % 2);
        }
        hostNumbers[i].len = (mbits) / 8;
        zapLeadingZeros(&hostNumbers[i]);
      }
      for (int i = OneMillion; i < hostNumbers.size(); ++i) {
        initBlkBU(&hostNumbers[i], 0);
      }

      sort(hostNumbers.begin(), hostNumbers.end(), weakCompareBU);

      GIVEN("an OpenCL buffer that can hold those numbers.") {
        int globalSize = nextPow2(hostNumbers.size());

        cl::Buffer buffer;
        REQUIRE(CLFW::get(buffer, "buffer", globalSize*sizeof(BigUnsigned)) == CL_SUCCESS);

        GIVEN("those numbers are uploaded sucessfully to the GPU") {
          REQUIRE(CLFW::DefaultQueue.enqueueWriteBuffer(buffer, true, 0, hostNumbers.size()*sizeof(BigUnsigned), hostNumbers.data()) == CL_SUCCESS);

          THEN("we can unique those numbers in parallel.") {
            cl_int newSize = hostNumbers.size();
            cl_int oldSize = hostNumbers.size();
            REQUIRE(UniqueSorted(buffer, newSize) == CL_SUCCESS);
            REQUIRE(newSize <= oldSize);
            
            AND_THEN("There are no race conditions.") {
              auto last = std::unique(hostNumbers.begin(), hostNumbers.end(), weakEqualsBU);
              hostNumbers.erase(last, hostNumbers.end());

              vector<BigUnsigned> GPUNumbers(newSize);
              REQUIRE(CLFW::DefaultQueue.enqueueReadBuffer(buffer, CL_TRUE, 0, newSize*sizeof(BigUnsigned), GPUNumbers.data()) == CL_SUCCESS);

              //Compare the serial calculations with the parallel calculations.
              REQUIRE(newSize == hostNumbers.size());
              int compareResult = 0;
              for (int i = 0; i < hostNumbers.size(); ++i) {
                compareResult = compareBU(&hostNumbers[i], &GPUNumbers[i]);
                if (compareResult != 0)
                  break;
              }
              REQUIRE(compareResult == 0);
            }
          }
        }
      }
    }
  }
}

SCENARIO("Sorted Z-Order numbers can be used to construct a binary radix tree") {
  cout << "Testing BuildBinaryRadixTree kernel" << endl;
  brtTestPassed = false;
  GIVEN("a fully initialized CLFW environment") {
    if (CLFW::IsNotInitialized()) REQUIRE(CLFW::Initialize() == CL_SUCCESS);

    /* initialize random seed: */
    srand(time(NULL));

    GIVEN("a couple sorted, uniqued points") {
      using namespace Kernels;
      vector<BigUnsigned> zpoints(nextPow2(OneMillion));
      for (int i = 0; i < OneMillion; ++i) {
        initBU(&zpoints[i]);
        for (int j = 0; j < (mbits); j++) {
          setBUBit(&zpoints[i], j, rand() % 2);
        }
        zpoints[i].len = (mbits) / 8;
        zapLeadingZeros(&zpoints[i]);
      }
      for (int i = OneMillion; i < zpoints.size(); ++i) {
        initBlkBU(&zpoints[i], 0);
      }

      sort(zpoints.begin(), zpoints.end(), weakCompareBU);

      auto last = unique(zpoints.begin(), zpoints.end(), weakEqualsBU);
      zpoints.erase(last, zpoints.end());

      GIVEN("an OpenCL buffer that can hold those numbers.") {
        using namespace Kernels;
        int globalSize = nextPow2(zpoints.size());

        //Create a BU buffer, but only if it hasn't been created in another test.
        cl::Buffer zpointsBuffer;
        REQUIRE(CLFW::get(zpointsBuffer, "zpointsBuffer", globalSize*sizeof(BigUnsigned)) == CL_SUCCESS);

        GIVEN("those points are then uploaded to the GPU") {
          REQUIRE(CLFW::DefaultQueue.enqueueWriteBuffer(zpointsBuffer, true, 0, zpoints.size()*sizeof(BigUnsigned), zpoints.data()) == CL_SUCCESS);

          THEN("we can build a BRT with those sorted, uniqued points in parallel") {
            REQUIRE(BuildBinaryRadixTree_p(zpointsBuffer, internalBRTNodes, zpoints.size(), mbits) == CL_SUCCESS);

            AND_THEN("we get no race conditions.") {
              //Determine what the BRT should look like
              vector<BrtNode> I(zpoints.size() - 1);
              REQUIRE(BuildBinaryRadixTree_s(zpoints.data(), I.data(), zpoints.size(), mbits) == CL_SUCCESS);

              //Fetch what the BRT actually looks like.
              vector<BrtNode> GPU_I(zpoints.size() - 1);
              REQUIRE(CLFW::DefaultQueue.enqueueReadBuffer(internalBRTNodes, CL_TRUE, 0, GPU_I.size()*sizeof(BrtNode), GPU_I.data()) == CL_SUCCESS);

              bool compareResult = true;
              for (int i = 0; i < I.size(); ++i) {
                compareResult = compareBrtNode(&GPU_I[i], &I[i]);
                if (compareResult == false) {
                  break;
                }
              }
              REQUIRE(compareResult == true);
              brtTestPassed = true;
              zpointsSize = zpoints.size();
            }
          }
        }
      }
    }
  }
}

SCENARIO("A binary radix tree can be used to construct a Octree/Quadtree") {
  cout << "Testing BinaryRadixToOctree kernel" << endl;
  //This isn't good practice, but it allows us to skip serializing BRT data.
  GIVEN("building the BRT was successful in the previous test.") {
    REQUIRE(brtTestPassed);
    THEN("we can use the BRT to build an Octree in parallel.") {
      using namespace Kernels;
      vector<OctNode> gpuOctree;
      REQUIRE(BinaryRadixToOctree_p(internalBRTNodes, gpuOctree, zpointsSize) == CL_SUCCESS);
      
      AND_THEN("we should get no race conditions.") {
        vector<OctNode> hostOctree;
        vector<BrtNode> I(zpointsSize-1);
        REQUIRE(CLFW::DefaultQueue.enqueueReadBuffer(internalBRTNodes, CL_TRUE, 0, (zpointsSize - 1)*sizeof(BrtNode), I.data())==CL_SUCCESS);

        REQUIRE(BinaryRadixToOctree_s(I, hostOctree, zpointsSize) == CL_SUCCESS);
        REQUIRE(hostOctree.size() == gpuOctree.size());

        bool compareResult = true;
        for (int i = 0; i < hostOctree.size(); ++i) {
          compareResult = compareOctNode(&gpuOctree[i], &hostOctree[i]);
          if (compareResult == false) {
            break;
          }
        }
        REQUIRE(compareResult == true);
      }
    }
  }
}
