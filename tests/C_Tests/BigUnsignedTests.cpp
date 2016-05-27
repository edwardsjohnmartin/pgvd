#include "catch.hpp"

#include "BigUnsigned.h"
#include "BigUnsigned.c"

using namespace std;
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

SCENARIO("BigUnsigneds can be constructed in multiple ways.") {
  GIVEN("an uninitialized BigUnsigned struct \"x\",") {
    BigUnsigned x;
    WHEN("initBU initializes the BU") {
      initBU(&x);
      THEN("the BU's length should equal 0.") {
        REQUIRE(x.len == 0);
      }
    }
    WHEN("initBlkBU initializes the BU with a number") {
      initBlkBU(&x, 42);
      THEN("the BU's length should equal 1.") {
        REQUIRE(x.len == 1);
      }
      THEN("the BU's first block should contain that number.") {
        REQUIRE(x.blk[0] == 42);
      }
    }
    WHEN("initBUBU initializes the BU with another initialized BU \"y\"") {
      BigUnsigned y;
      y.len = BIG_INTEGER_SIZE;
      for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
        y.blk[i] = i;
      }
      initBUBU(&x, &y);
      THEN("x's length should match y's length.") 
        REQUIRE(x.len == y.len);
      THEN("x's blocks should match y's blocks.") {
        for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
          CAPTURE(i);
          REQUIRE(x.blk[i] == y.blk[i]);
        }
      }
    }
  }
}
SCENARIO("BigUnsigneds can be added.") {
  GIVEN("A BU \"x\" containing the value " + to_string((int)pow(2, (sizeof(Blk) * 8))-1)) {
    BigUnsigned x;
    initBlkBU(&x, (unsigned char)pow(2, (sizeof(Blk) * 8))-1);
    WHEN("we add another BU \"y\" containing the value 1") {
      BigUnsigned y;
      BigUnsigned z;
      initBlkBU(&y, 1);
      addBU(&x, &x, &y);
      THEN("the overflow should carry over, causing the BU \"x\" to have a length of 2.") {
        REQUIRE(x.len == 2);
      }
      THEN("the overflow should carry over, causing the BU \"x\" to contain the value [1][0]") {
        REQUIRE(x.blk[1] == 1);
        REQUIRE(x.blk[0] == 0);
      }
    }
  }
  GIVEN("Two fully filled BUs") {
    BigUnsigned x;
    x.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      x.blk[i] = i;
    }
    BigUnsigned y;
    y.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      y.blk[i] = i+1;
    }
    INFO( "x's contents: " + buToString( x ) );
    INFO( "y's contents: " + buToString( y ) );
    WHEN("those two BU's are added together") {
      BigUnsigned z;
      addBU(&z, &x, &y);
      INFO("z's contents: " + buToString(z));
      THEN("each block in the same column should be added together.") {
        for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
          REQUIRE(z.blk[i] == x.blk[i] + y.blk[i]);
        }
      }
    }
  }
}
SCENARIO("BigUnsigneds can be subtracted.") {
  GIVEN("A BU \"x\" containing the value [1][0]") {
    BigUnsigned x;
    x.len = 2;
    x.blk[0] = 0;
    x.blk[1] = 1;
    WHEN("we subtract another BU \"y\" containing the value [1]") {
      BigUnsigned y;
      BigUnsigned z;
      initBlkBU(&y, 1);
      subtractBU(&x, &x, &y);
      THEN("a number will be borrowed from the next significant digit, causing the BU \"x\" to have a length of 1.") {
        REQUIRE(x.len == 1);
      }
      THEN("a number will be borrowed from the next significant digit, causing the BU \"x\" to contain the value [0]" + to_string((int)pow(2,sizeof(Blk)*8)-1)) {
        REQUIRE(x.blk[0] == pow(2, sizeof(Blk) * 8)-1);
      }
    }
  }
  GIVEN("Two fully filled BUs") {
    BigUnsigned x;
    x.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      x.blk[i] = i+1;
    }
    BigUnsigned y;
    y.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      y.blk[i] = i;
    }
    INFO("x's contents: " + buToString(x));
    INFO("y's contents: " + buToString(y));
    WHEN("those two BU's are subtracted from eachother") {
      BigUnsigned z;
      subtractBU(&z, &x, &y);
      INFO("z's contents: " + buToString(z));
      THEN("each block in the same column should be subtracted from eachother.") {
        for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
          REQUIRE(z.blk[i] == x.blk[i] - y.blk[i]);
        }
      }
    }
  }
}
SCENARIO("BigUnsigneds can be &'d together.") { 
  GIVEN("Two fully filled BUs") {
    BigUnsigned x;
    x.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      x.blk[i] = i + 1;
    }
    BigUnsigned y;
    y.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      y.blk[i] = i;
    }
    INFO("x's contents: " + buToString(x));
    INFO("y's contents: " + buToString(y));
    WHEN("those two BigUnsigneds are &'d together") {
      BigUnsigned z;
      andBU(&z, &x, &y);
      THEN("each block is column wise &'d together") {
        for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
          CAPTURE(i);
          REQUIRE(z.blk[i] == (x.blk[i] & y.blk[i]));
        }
      }
    }
  }
}
SCENARIO("BigUnsigneds can be |'d together.") { 
  GIVEN("Two fully filled BUs") {
    BigUnsigned x;
    x.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      x.blk[i] = i + 1;
    }
    BigUnsigned y;
    y.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      CAPTURE(i);
      y.blk[i] = i;
    }
    INFO("x's contents: " + buToString(x));
    INFO("y's contents: " + buToString(y));
    WHEN("those two BigUnsigneds are |'d together") {
      BigUnsigned z;
      orBU(&z, &x, &y);
      THEN("each block is column wise |'d together") {
        for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
          CAPTURE(i);
          REQUIRE(z.blk[i] == (x.blk[i] | y.blk[i]));
        }
      }
    }
  }
}
SCENARIO("BigUnsigneds can be ^'d together.") { 
  GIVEN("Two fully filled BUs") {
    BigUnsigned x;
    x.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      x.blk[i] = i + 1;
    }
    BigUnsigned y;
    y.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      y.blk[i] = i;
    }
    INFO("x's contents: " + buToString(x));
    INFO("y's contents: " + buToString(y));
    WHEN("those two BigUnsigneds are ^'d together") {
      BigUnsigned z;
      xOrBU(&z, &x, &y);
      THEN("each block is column wise ^'d together") {
        for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
          REQUIRE(z.blk[i] == (x.blk[i] ^ y.blk[i]));
        }
      }
    }
  }
}
SCENARIO("BigUnsigneds can be >>'d.") { 
  GIVEN("a BU of any block size, say " + to_string(sizeof(Blk) * 8) + " bits.") {
    WHEN("a BU contains a value, say [3][2]") {
      BigUnsigned x;
      x.len = 2;
      x.blk[0] = 2;
      x.blk[1] = 3;
      AND_WHEN("that BU is >>'d by a number, say 1") {
        shiftBURight(&x, &x, 1);
        INFO("X's contents: " + buToString(x));
        THEN("bits shift from one block to another.") {
          REQUIRE(x.blk[0] == (1 << (sizeof(Blk) * 7)) + 1);
          REQUIRE(x.blk[1] == 1);
        }
      }
    }
  }
}
SCENARIO("BigUnsigneds can be <<'d.") {
  GIVEN("a BU of any block size, say " + to_string(sizeof(Blk)*8) + " bits.") {
    WHEN("that BU contains a number, say " + to_string((1<<(sizeof(Blk)*7))+1)) {
      BigUnsigned x;
      x.len = 1;
      x.blk[0] = (1 << (sizeof(Blk) * 7)) + 1;
      AND_WHEN("that BU is <<'d by a number, say 1") {
        shiftBULeft(&x, &x, 1);
        INFO("X's contents: " + buToString(x));
        THEN("bits shift from one block to another.") {
          REQUIRE(x.blk[0] == 2);
          REQUIRE(x.blk[1] == 1);
        }
      }
    }
  }
}
SCENARIO("BigUnsigneds can be compared.") {
  GIVEN("Two fully filled BUs, \"x\" containing a larger number than the other") {
    BigUnsigned x;
    x.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      x.blk[i] = i + 1;
    }
    BigUnsigned y;
    y.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      y.blk[i] = i;
    }
    INFO("x's contents: " + buToString(x));
    INFO("y's contents: " + buToString(y));
    WHEN("those BUs are compared") {
      THEN("the comparison with x on the left returns 1, meaning x is greater.") {
        REQUIRE(compareBU(&x, &y) == 1);
      }
      THEN("the comparison with x on the right returns -1, meaning y is smaller.") {
        REQUIRE(compareBU(&y, &x) == -1);
      }
    }
  }
  GIVEN("Two fully filled BUs containing the same number") {
    BigUnsigned x;
    x.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      x.blk[i] = i;
    }
    BigUnsigned y;
    y.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      y.blk[i] = i;
    }
    INFO("x's contents: " + buToString(x));
    INFO("y's contents: " + buToString(y));
    WHEN("those BUs are compared") {
      THEN("the comparison with x on the left returns 0, meaning the BUs are equal.") {
        REQUIRE(compareBU(&x, &y) == 0);
      }
      THEN("the comparison with x on the right returns 0, meaning the BUs are equal.") {
        REQUIRE(compareBU(&y, &x) == 0);
      }
    }
  }
}
SCENARIO("Bits and blocks within a BigUnsigned can be accessed in a couple ways.") {
  GIVEN("a fully filled BU") {
    BigUnsigned x;
    x.len = BIG_INTEGER_SIZE;
    for (int i = 0; i < BIG_INTEGER_SIZE; ++i) {
      x.blk[i] = i;
    }
    THEN("a block within that BU can be accessed.") {
      REQUIRE(getBUBlock(&x, 3) == 3);
    }
    THEN("a block within that BU can be set.") {
      setBUBlock(&x, 2, 42);
      INFO("x's contents: " + buToString(x));
      REQUIRE(x.blk[2] == 42);
    }
    THEN("a shifted block within that BU can be accessed.") {
      REQUIRE(getShiftedBUBlock(&x, 1, 1) == 2);
    }
    THEN("a bit within a BU should be accessable.") {
      REQUIRE(getBUBit(&x, sizeof(Blk)*8) == 1);
    }
    THEN("a bit within a BU can be set.") {
      setBUBit(&x, 1, 1);
      INFO("x's contents: " + buToString(x));
      REQUIRE(x.blk[0] == 2);
    }
  }
}
