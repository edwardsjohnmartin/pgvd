#include "Catch/catch.hpp"
#include "BigUnsigned/BigNum.h"

#if NumBlocks == 4
#define longMax 18446744073709551615
Scenario("bigs can be added together", "[big]") {
	Given("two bigs") {
		cl_ulong test = 0;
		big x = { 1, 1, 1, 0};
		big y = {0, longMax, 1, 0};
		big z = addBig(&x, &y);
		Require(z.blk[0] == 1);
		Require(z.blk[1] == 0);
		Require(z.blk[2] == 3);
		Require(z.blk[3] == 0);
	}
}

Scenario("bigs can be subtracted from each other", "[big]") {
	Given("two bigs") {
		cl_ulong test = 0;
		big x = { 1, 0, 3, 0 };
		big y = { 1, 1, 1, 0 };
		big z = subtractBig(&x, &y);
		Require(z.blk[0] == 0);
		Require(z.blk[1] == longMax);
		Require(z.blk[2] == 1);
		Require(z.blk[3] == 0);
	}
}

Scenario("bigs can be & together", "[big]") {
	Given("two bigs") {
		cl_ulong test = 0;
		big x = { 3, 3, 3, 3 };
		big y = { 0, 1, 2, 3 };
		big z = andBig(&x, &y);
		Require(z.blk[0] == 0);
		Require(z.blk[1] == 1);
		Require(z.blk[2] == 2);
		Require(z.blk[3] == 3);
	}
}

Scenario("bigs can be | together", "[big]") {
	Given("two bigs") {
		cl_ulong test = 0;
		big x = { 3, 2, 1, 0 };
		big y = { 0, 1, 2, 0 };
		big z = orBig(&x, &y);
		Require(z.blk[0] == 3);
		Require(z.blk[1] == 3);
		Require(z.blk[2] == 3);
		Require(z.blk[3] == 0);
	}
}

Scenario("bigs can be ^ together", "[big]") {
	Given("two bigs") {
		cl_ulong test = 0;
		big x = { 3, 2, 1, 0 };
		big y = { 0, 3, 2, 0 };
		big z = xOrBig(&x, &y);
		Require(z.blk[0] == 3);
		Require(z.blk[1] == 1);
		Require(z.blk[2] == 3);
		Require(z.blk[3] == 0);
	}
}

Scenario("bigs can be >>", "[big]") {
	Given("a big and a positive number") {
		big x = { 1, 2, 3, 4 };
		big y = shiftBigRight(&x, 64 + 1);
		Require(y.blk[3] == 0);
		Require(y.blk[2] == (4ull >> 1));
		Require(y.blk[1] == (3UL >> 1));
		Require(y.blk[0] == 9223372036854775809);
	}
}

Scenario("bigs can be <<", "[big]") {
	Given("a big and a positive number") {
		big x = { 1ull << 63, 1, 3, 3 };
		big y = shiftBigLeft(&x, 64 + 1);
		Require(y.blk[0] == 0);
		Require(y.blk[1] == 0);
		Require(y.blk[2] == 3);
		Require(y.blk[3] == 6);
	}
}
#endif
