#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "clfw.hpp"
#include <cstdlib>
extern "C" {
#include "BinaryRadixTree/BuildBRT.h"
}

int main( int argc, char* const argv[] )
{
  /* global setup... */
  CLFW::Initialize(true, true, 2);

  int result = Catch::Session().run( argc, argv );
  
  /* global clean-up... */
  system("pause");
  return result;
}
