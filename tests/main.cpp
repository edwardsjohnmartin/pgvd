#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "BigUnsigned.h"
#include "clfw.hpp"
#include "catch.hpp"

int main(int argc, char* const argv[]) {
  int result = -1;
  // global setup...
  //if (CLFW::Initialize(true) == 0) {
    result = Catch::Session().run(argc, argv);

    // global clean-up...
    //CLFW::Terminate();
 // }
  return result;
}