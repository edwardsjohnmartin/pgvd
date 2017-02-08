#include "Kernels/Kernels.h"
#include "HelperFunctions.hpp"
#include "catch.hpp"
using namespace cl;
using namespace Kernels;
using namespace GLUtilities;

static inline std::string BuToString(BigUnsigned bu) {
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
static void clearScreen() {
	using namespace GLUtilities;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
static void refresh() {
	using namespace GLUtilities;
	Sketcher::instance()->draw();
	glfwSwapBuffers(GLUtilities::window);
}

/* Reduction Kernels */
Scenario("Additive Reduction", "[reduction]") {
	Given("N random integers") {
		vector<cl_int>small_input = generateDeterministicRandomIntegers(a_few);
    vector<cl_int>large_input(a_lot, 1);// = generateDeterministicRandomIntegers(a_lot, );
		When("we reduce these numbers in series") {
			int small_output_s, large_output_s;
			Reduce_s(small_input, small_output_s);
			Reduce_s(large_input, large_output_s);
			Then("we get the summation of those integers") {
        cl_int small_actual = 0;
        cl_int large_actual = 0;
        for (int i = 0; i < a_few; ++i)
          small_actual += small_input[i];
        for (int i = 0; i < a_lot; ++i)
          large_actual += large_input[i];
				Require(small_output_s == small_actual);
				Require(large_output_s == large_actual);
			}
			Then("the series results match the parallel results") {
				cl_int small_output_p, large_output_p;
				Buffer b_small_input, b_large_input, b_small_output, b_large_output;
				cl_int error = 0;
				error |= CLFW::get(b_small_input, "small", a_few * sizeof(cl_int));
				error |= CLFW::get(b_large_input, "large", a_lot * sizeof(cl_int));
				error |= CLFW::Upload<cl_int>(small_input, b_small_input);
				error |= CLFW::Upload<cl_int>(large_input, b_large_input);
				Require(error == CL_SUCCESS);
				error |= Reduce_p(b_small_input, a_few, "a", b_small_output);
				error |= Reduce_p(b_large_input, a_lot, "b", b_large_output);
				error |= CLFW::Download<cl_int>(b_small_output, 0, small_output_p);
				error |= CLFW::Download<cl_int>(b_large_output, 0, large_output_p);
				Require(small_output_p == small_output_s);
				Require(large_output_p == large_output_s);
			}
		}
	}
}

/* Predication Kernels */
Scenario("Predicate by bit", "[predication]") {
	Given("N random integers") {
		vector<cl_int>small_input = generateDeterministicRandomIntegers(a_few);
		vector<cl_int>large_input = generateDeterministicRandomIntegers(a_lot);
		When("we predicate these integers in series") {
			vector<cl_int>small_output_s(a_few), large_output_s(a_lot);
			PredicateByBit_s(small_input, 2, 1, small_output_s);
			PredicateByBit_s(large_input, 3, 0, large_output_s);
			Then("the predication for each number will be 1 only if the i'th bit matches \"compared\"") {
				int success = true;
				for (int i = 0; i < a_few; ++i)
					success &= ((small_input[i] & (1 << 2)) >> 2 == 1) == small_output_s[i];
				for (int i = 0; i < a_lot; ++i)
					success &= ((large_input[i] & (1 << 3)) >> 3 == 0) == large_output_s[i];
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				vector<cl_int>small_output_p(a_few);
				vector<cl_int>large_output_p(a_lot);
				Buffer b_small_input, b_large_input, b_small_output, b_large_output;
				cl_int error = 0;
				error |= CLFW::get(b_small_input, "small", a_few * sizeof(cl_int));
				error |= CLFW::get(b_large_input, "large", a_lot * sizeof(cl_int));
				error |= CLFW::Upload<cl_int>(small_input, b_small_input);
				error |= CLFW::Upload<cl_int>(large_input, b_large_input);
				error |= PredicateByBit_p(b_small_input, 2, 1, a_few, "a", b_small_output);
				error |= PredicateByBit_p(b_large_input, 3, 0, a_lot, "b", b_large_output);
				error |= CLFW::Download<cl_int>(b_small_output, a_few, small_output_p);
				error |= CLFW::Download<cl_int>(b_large_output, a_lot, large_output_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i) success &= small_output_p[i] == small_output_s[i];
				for (int i = 0; i < a_lot; ++i) success &= large_output_p[i] == large_output_s[i];
				Require(success == true);
			}
		}
	}
}
Scenario("Predicate BigUnsigned by bit", "[predication]") {
	Given("N random BigUnsigneds") {
		vector<BigUnsigned>small_input = generateDeterministicRandomBigUnsigneds(a_few);
		vector<BigUnsigned>large_input = generateDeterministicRandomBigUnsigneds(a_lot);
		When("we predicate these integers in series") {
			vector<cl_int>small_output_s(a_few), large_output_s(a_lot);
			PredicateBUByBit_s(small_input, 2, 1, small_output_s);
			PredicateBUByBit_s(large_input, 3, 0, large_output_s);
			Then("the predication for each number will be 1 only if the i'th bit matches \"compared\"") {
				int success = true;
				for (int i = 0; i < a_few; ++i)
					success &= (getBUBit(&small_input[i], 2) == 1) == small_output_s[i];
				for (int i = 0; i < a_lot; ++i)
					success &= (getBUBit(&large_input[i], 3) == 0) == large_output_s[i];
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				vector<cl_int>small_output_p(a_few);
				vector<cl_int>large_output_p(a_lot);
				Buffer b_small_input, b_large_input, b_small_output, b_large_output;
				cl_int error = 0;
				error |= CLFW::get(b_small_input, "small", a_few * sizeof(BigUnsigned));
				error |= CLFW::get(b_large_input, "large", a_lot * sizeof(BigUnsigned));
				error |= CLFW::Upload<BigUnsigned>(small_input, b_small_input);
				error |= CLFW::Upload<BigUnsigned>(large_input, b_large_input);
				error |= PredicateBUByBit_p(b_small_input, 2, 1, a_few, "a", b_small_output);
				error |= PredicateBUByBit_p(b_large_input, 3, 0, a_lot, "b", b_large_output);
				error |= CLFW::Download<cl_int>(b_small_output, a_few, small_output_p);
				error |= CLFW::Download<cl_int>(b_large_output, a_lot, large_output_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i) success &= small_output_p[i] == small_output_s[i];
				for (int i = 0; i < a_lot; ++i) success &= large_output_p[i] == large_output_s[i];
				Require(success == true);
			}
		}
	}
}
Scenario("Predicate Conflict", "[conflict][predication]") {
	Given("N conflicts") {
		vector<Conflict>small_input = generateDeterministicRandomConflicts(a_few);
		vector<Conflict>large_input = generateDeterministicRandomConflicts(a_lot);
		When("we predicate these conflicts in series") {
			vector<cl_int>small_output_s(a_few), large_output_s(a_lot);
			PredicateConflicts_s(small_input, small_output_s);
			PredicateConflicts_s(large_input, large_output_s);
			Then("the predication for each number will be 1 only if the i'th bit matches \"compared\"") {
				int success = true;
				for (int i = 0; i < a_few; ++i)
					success &= (small_input[i].color == -2) == small_output_s[i];
				for (int i = 0; i < a_lot; ++i)
					success &= (large_input[i].color == -2) == large_output_s[i];
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				vector<cl_int>small_output_p(a_few);
				vector<cl_int>large_output_p(a_lot);
				Buffer b_small_input, b_large_input, b_small_output, b_large_output;
				cl_int error = 0;
				error |= CLFW::get(b_small_input, "small", a_few * sizeof(Conflict));
				error |= CLFW::get(b_large_input, "large", a_lot * sizeof(Conflict));
				error |= CLFW::Upload<Conflict>(small_input, b_small_input);
				error |= CLFW::Upload<Conflict>(large_input, b_large_input);
				error |= PredicateConflicts_p(b_small_input, a_few, "a", b_small_output);
				error |= PredicateConflicts_p(b_large_input, a_lot, "b", b_large_output);
				error |= CLFW::Download<cl_int>(b_small_output, a_few, small_output_p);
				error |= CLFW::Download<cl_int>(b_large_output, a_lot, large_output_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i) success &= small_output_p[i] == small_output_s[i];
				for (int i = 0; i < a_lot; ++i) success &= large_output_p[i] == large_output_s[i];
				Require(success == true);
			}
		}
	}
}

/* Compaction Kernels */
Scenario("Integer Compaction", "[compaction]") {
	Given("N random integers, an arbitrary predication, and an inclusive prefix sum of that predication") {
		vector<cl_int>small_input = generateDeterministicRandomIntegers(a_few);
		vector<cl_int>large_input = generateDeterministicRandomIntegers(a_lot);
		vector<cl_int> small_pred(a_few), small_addr(a_few), small_output_s(a_few);
		vector<cl_int> large_pred(a_lot), large_addr(a_lot), large_output_s(a_lot);
		/* In this example, odd indexes are compacted to the left. */
		for (int i = 0; i < a_few; ++i) { small_pred[i] = i % 2; small_addr[i] = (i + 1) / 2; }
		for (int i = 0; i < a_lot; ++i) { large_pred[i] = i % 2; large_addr[i] = (i + 1) / 2; }
		When("these integers are compacted in series") {
			Compact_s(small_input, small_pred, small_addr, small_output_s);
			Compact_s(large_input, large_pred, large_addr, large_output_s);
			Then("elements predicated true are moved to their cooresponding addresses.") {
				int success = true;
				for (int i = 0; i < a_few / 2; ++i) success &= (small_output_s[i] == small_input[(i * 2) + 1]);
				for (int i = 0; i < a_lot / 2; ++i) success &= (large_output_s[i] == large_input[(i * 2) + 1]);
				Require(success == true);
			}
			Then("elements predicated false are placed after the last truely predicated element and in their original order.") {
				int success = true;
				for (int i = a_few / 2; i < a_few; ++i) success &= (small_output_s[i] == small_input[(i - (a_few / 2)) * 2]);
				for (int i = a_lot / 2; i < a_lot; ++i) success &= (large_output_s[i] == large_input[(i - (a_lot / 2)) * 2]);
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<cl_int> small_output_p(a_few);
				vector<cl_int> large_output_p(a_lot);
				Buffer b_small_input, b_small_pred, b_small_addr, b_small_output;
				Buffer b_large_input, b_large_pred, b_large_addr, b_large_output;
				error |= CLFW::get(b_small_input, "b_small_input", a_few * sizeof(cl_int));
				error |= CLFW::get(b_small_pred, "b_small_pred", a_few * sizeof(cl_int));
				error |= CLFW::get(b_small_addr, "b_small_addr", a_few * sizeof(cl_int));
				error |= CLFW::get(b_small_output, "b_small_output", a_few * sizeof(cl_int));
				error |= CLFW::get(b_large_input, "b_large_input", a_lot * sizeof(cl_int));
				error |= CLFW::get(b_large_pred, "b_large_pred", a_lot * sizeof(cl_int));
				error |= CLFW::get(b_large_addr, "b_large_addr", a_lot * sizeof(cl_int));
				error |= CLFW::get(b_large_output, "b_large_output", a_lot * sizeof(cl_int));
				error |= CLFW::Upload<cl_int>(small_input, b_small_input);
				error |= CLFW::Upload<cl_int>(small_pred, b_small_pred);
				error |= CLFW::Upload<cl_int>(small_addr, b_small_addr);
				error |= CLFW::Upload<cl_int>(large_input, b_large_input);
				error |= CLFW::Upload<cl_int>(large_pred, b_large_pred);
				error |= CLFW::Upload<cl_int>(large_addr, b_large_addr);
				error |= Compact_p(b_small_input, b_small_pred, b_small_addr, a_few, b_small_output);
				error |= Compact_p(b_large_input, b_large_pred, b_large_addr, a_lot, b_large_output);
				error |= CLFW::Download<cl_int>(b_small_output, a_few, small_output_p);
				error |= CLFW::Download<cl_int>(b_large_output, a_lot, large_output_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i) success &= (small_output_p[i] == small_output_s[i]);
				for (int i = 0; i < a_lot; ++i) success &= (large_output_p[i] == large_output_s[i]);
				Require(success == true);
			}
		}
	}
}
Scenario("Big Unsigned Compaction", "[compaction]") {
	Given("N random BigUnsigneds, an arbitrary predication, and an inclusive prefix sum of that predication") {
		vector<BigUnsigned>small_input = generateDeterministicRandomBigUnsigneds(a_few);
		vector<BigUnsigned>large_input = generateDeterministicRandomBigUnsigneds(a_lot);
		vector<cl_int> small_pred(a_few), small_addr(a_few);
		vector<cl_int> large_pred(a_lot), large_addr(a_lot);
		vector<BigUnsigned> small_output_s(a_few), large_output_s(a_lot);
		/* In this example, odd indexes are compacted to the left. */
		for (int i = 0; i < a_few; ++i) { small_pred[i] = i % 2; small_addr[i] = (i + 1) / 2; }
		for (int i = 0; i < a_lot; ++i) { large_pred[i] = i % 2; large_addr[i] = (i + 1) / 2; }
		When("these integers are compacted in series") {
			BUCompact_s(small_input, small_pred, small_addr, small_output_s);
			BUCompact_s(large_input, large_pred, large_addr, large_output_s);
			Then("elements predicated true are moved to their cooresponding addresses.") {
				int success = true;
				for (int i = 0; i < a_few / 2; ++i) success &= (compareBU(&small_output_s[i], &small_input[(i * 2) + 1]) == 0);
				for (int i = 0; i < a_lot / 2; ++i) success &= (compareBU(&large_output_s[i], &large_input[(i * 2) + 1]) == 0);
				Require(success == true);
			}
			Then("elements predicated false are placed after the last truely predicated element and in their original order.") {
				int success = true;
				for (int i = a_few / 2; i < a_few; ++i) success &= (compareBU(&small_output_s[i], &small_input[(i - (a_few / 2)) * 2]) == 0);
				for (int i = a_lot / 2; i < a_lot; ++i) success &= (compareBU(&large_output_s[i], &large_input[(i - (a_lot / 2)) * 2]) == 0);
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<BigUnsigned> small_output_p(a_few);
				vector<BigUnsigned> large_output_p(a_lot);
				Buffer b_small_input, b_small_pred, b_small_addr, b_small_output;
				Buffer b_large_input, b_large_pred, b_large_addr, b_large_output;
				error |= CLFW::get(b_small_input, "b_small_input", a_few * sizeof(BigUnsigned));
				error |= CLFW::get(b_small_pred, "b_small_pred", a_few * sizeof(cl_int));
				error |= CLFW::get(b_small_addr, "b_small_addr", a_few * sizeof(cl_int));
				error |= CLFW::get(b_small_output, "b_small_output", a_few * sizeof(BigUnsigned));
				error |= CLFW::get(b_large_input, "b_large_input", a_lot * sizeof(BigUnsigned));
				error |= CLFW::get(b_large_pred, "b_large_pred", a_lot * sizeof(cl_int));
				error |= CLFW::get(b_large_addr, "b_large_addr", a_lot * sizeof(cl_int));
				error |= CLFW::get(b_large_output, "b_large_output", a_lot * sizeof(BigUnsigned));
				error |= CLFW::Upload<BigUnsigned>(small_input, b_small_input);
				error |= CLFW::Upload<cl_int>(small_pred, b_small_pred);
				error |= CLFW::Upload<cl_int>(small_addr, b_small_addr);
				error |= CLFW::Upload<BigUnsigned>(large_input, b_large_input);
				error |= CLFW::Upload<cl_int>(large_pred, b_large_pred);
				error |= CLFW::Upload<cl_int>(large_addr, b_large_addr);
				error |= BUCompact_p(b_small_input, a_few, b_small_pred, b_small_addr, b_small_output);
				error |= BUCompact_p(b_large_input, a_lot, b_large_pred, b_large_addr, b_large_output);
				error |= CLFW::Download<BigUnsigned>(b_small_output, a_few, small_output_p);
				error |= CLFW::Download<BigUnsigned>(b_large_output, a_lot, large_output_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i) success &= (compareBU(&small_output_p[i], &small_output_s[i]) == 0);
				for (int i = 0; i < a_lot; ++i) success &= (compareBU(&large_output_p[i], &large_output_s[i]) == 0);
				Require(success == true);
			}
		}
	}
}
Scenario("Conflict Compaction", "[conflict][compaction]") {
	Given("N random conflicts, an arbitrary predication, and an inclusive prefix sum of that predication") {
		vector<Conflict>small_input = generateDeterministicRandomConflicts(a_few);
		vector<Conflict>large_input = generateDeterministicRandomConflicts(a_lot);
		vector<cl_int> small_pred(a_few), small_addr(a_few);
		vector<cl_int> large_pred(a_lot), large_addr(a_lot);
		vector<Conflict>small_output_s(a_few), large_output_s(a_lot);
		/* In this example, odd indexes are compacted to the left. */
		for (int i = 0; i < a_few; ++i) { small_pred[i] = i % 2; small_addr[i] = (i + 1) / 2; }
		for (int i = 0; i < a_lot; ++i) { large_pred[i] = i % 2; large_addr[i] = (i + 1) / 2; }
		When("these conflicts are compacted in series") {
			CompactConflicts_s(small_input, small_pred, small_addr, small_output_s);
			CompactConflicts_s(large_input, large_pred, large_addr, large_output_s);
			Then("elements predicated true are moved to their cooresponding addresses.") {
				int success = true;
				for (int i = 0; i < a_few / 2; ++i) success &= (compareConflict(&small_output_s[i], &small_input[(i * 2) + 1]));
				for (int i = 0; i < a_lot / 2; ++i) success &= (compareConflict(&large_output_s[i], &large_input[(i * 2) + 1]));
				Require(success == true);
			}
			Then("elements predicated false are placed after the last truely predicated element and in their original order.") {
				int success = true;
				for (int i = a_few / 2; i < a_few; ++i) success &= (compareConflict(&small_output_s[i], &small_input[(i - (a_few / 2)) * 2]));
				for (int i = a_lot / 2; i < a_lot; ++i) success &= (compareConflict(&large_output_s[i], &large_input[(i - (a_lot / 2)) * 2]));
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<Conflict> small_output_p(a_few);
				vector<Conflict> large_output_p(a_lot);
				Buffer b_small_input, b_small_pred, b_small_addr, b_small_output;
				Buffer b_large_input, b_large_pred, b_large_addr, b_large_output;
				error |= CLFW::get(b_small_input, "b_small_input", a_few * sizeof(Conflict));
				error |= CLFW::get(b_small_pred, "b_small_pred", a_few * sizeof(cl_int));
				error |= CLFW::get(b_small_addr, "b_small_addr", a_few * sizeof(cl_int));
				error |= CLFW::get(b_small_output, "b_small_output", a_few * sizeof(Conflict));
				error |= CLFW::get(b_large_input, "b_large_input", a_lot * sizeof(Conflict));
				error |= CLFW::get(b_large_pred, "b_large_pred", a_lot * sizeof(cl_int));
				error |= CLFW::get(b_large_addr, "b_large_addr", a_lot * sizeof(cl_int));
				error |= CLFW::get(b_large_output, "b_large_output", a_lot * sizeof(Conflict));
				error |= CLFW::Upload<Conflict>(small_input, b_small_input);
				error |= CLFW::Upload<cl_int>(small_pred, b_small_pred);
				error |= CLFW::Upload<cl_int>(small_addr, b_small_addr);
				error |= CLFW::Upload<Conflict>(large_input, b_large_input);
				error |= CLFW::Upload<cl_int>(large_pred, b_large_pred);
				error |= CLFW::Upload<cl_int>(large_addr, b_large_addr);
				error |= CompactConflicts_p(b_small_input, b_small_pred, b_small_addr, a_few, b_small_output);
				error |= CompactConflicts_p(b_large_input, b_large_pred, b_large_addr, a_lot, b_large_output);
				error |= CLFW::Download<Conflict>(b_small_output, a_few, small_output_p);
				error |= CLFW::Download<Conflict>(b_large_output, a_lot, large_output_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i) success &= (compareConflict(&small_output_p[i], &small_output_s[i]));
				for (int i = 0; i < a_lot; ++i) success &= (compareConflict(&large_output_p[i], &large_output_s[i]));
				Require(success == true);
			}
		}
	}
}

/* Scan Kernels */
Scenario("Inclusive Summation Scan", "[scan]") {
	Given("N random integers") {
		vector<cl_int>small_input = generateDeterministicRandomIntegers(a_few);
		vector<cl_int>large_input = generateDeterministicRandomIntegers(a_lot);
		When("we scan these integers in series") {
			vector<cl_int>small_output_s(a_few), large_output_s(a_lot);
			StreamScan_s(small_input, small_output_s);
			StreamScan_s(large_input, large_output_s);
			Then("the predication for each number will be 1 only if the i'th bit matches \"compared\"") {
				int success = true;
				Require(small_input[0] == small_output_s[0]);
				for (int i = 1; i < a_few; ++i)
					success &= (small_output_s[i] == small_output_s[i - 1] + small_input[i]);
				Require(large_input[0] == large_output_s[0]);
				for (int i = 1; i < a_lot; ++i)
					success &= (large_output_s[i] == large_output_s[i - 1] + large_input[i]);
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				vector<cl_int>small_output_p(a_few), large_output_p(a_lot);
				Buffer b_small_input, b_large_input, b_small_output, b_large_output;
				cl_int error = 0;
				error |= CLFW::get(b_small_input, "smallin", a_few * sizeof(cl_int));
				error |= CLFW::get(b_large_input, "largein", a_lot * sizeof(cl_int));
				error |= CLFW::get(b_small_output, "smallout", a_few * sizeof(cl_int));
				error |= CLFW::get(b_large_output, "largeout", a_lot * sizeof(cl_int));
				error |= CLFW::Upload<cl_int>(small_input, b_small_input);
				error |= CLFW::Upload<cl_int>(large_input, b_large_input);
				error |= StreamScan_p(b_small_input, a_few, "a", b_small_output);
				error |= CLFW::DefaultQueue.finish();
				error |= StreamScan_p(b_large_input, a_lot, "b", b_large_output);
				error |= CLFW::DefaultQueue.finish();
				error |= CLFW::Download<cl_int>(b_small_output, a_few, small_output_p);
				error |= CLFW::Download<cl_int>(b_large_output, a_lot, large_output_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i)
					success &= small_output_p[i] == small_output_s[i];
				for (int i = 0; i < a_lot; ++i)
					success &= large_output_p[i] == large_output_s[i];
				Require(success == true);
			}
		}
	}
}

/* Sort Routines */
Scenario("Parallel Radix Sort (Pairs by Key)", "[sort][integration]") {
	Given("An arbitrary set of unsigned key and integer value pairs") {
		vector<cl_int> small_keys_in(a_few);
		vector<cl_int> small_values_in(a_few);
		vector<cl_int> large_keys_in(a_lot);
		vector<cl_int> large_values_in(a_lot);

		for (int i = 0; i < a_few; ++i) small_keys_in[i] = small_values_in[i] = a_few - i;
		for (int i = 0; i < a_lot; ++i) large_keys_in[i] = large_values_in[i] = a_lot - i;

		When("these pairs are sorted by key in parallel") {
			cl_int error = 0;
			cl::Buffer b_small_keys, b_small_values, b_large_keys, b_large_values;
			error |= CLFW::get(b_small_keys, "b_small_keys", a_few * sizeof(cl_int));
			error |= CLFW::get(b_small_values, "b_small_values", a_few * sizeof(cl_int));
			error |= CLFW::get(b_large_keys, "b_large_keys", a_lot * sizeof(cl_int));
			error |= CLFW::get(b_large_values, "b_large_values", a_lot * sizeof(cl_int));
			error |= CLFW::Upload<cl_int>(small_keys_in, b_small_keys);
			error |= CLFW::Upload<cl_int>(small_values_in, b_small_values);
			error |= CLFW::Upload<cl_int>(large_keys_in, b_large_keys);
			error |= CLFW::Upload<cl_int>(large_values_in, b_large_values);
			Require(error == CL_SUCCESS);
			error |= RadixSortPairsByKey(b_small_keys, b_small_values, a_few);
			Require(error == CL_SUCCESS);
			error |= RadixSortPairsByKey(b_large_keys, b_large_values, a_lot);
			Require(error == CL_SUCCESS);
			Then("The key value pairs are ordered by keys assending") {
				vector<cl_int> small_keys_out_p(a_few), small_values_out_p(a_few);
				vector<cl_int> large_keys_out_p(a_lot), large_values_out_p(a_lot);
				error |= CLFW::Download<cl_int>(b_small_keys, a_few, small_keys_out_p);
				error |= CLFW::Download<cl_int>(b_small_values, a_few, small_values_out_p);
				error |= CLFW::Download<cl_int>(b_large_keys, a_lot, large_keys_out_p);
				error |= CLFW::Download<cl_int>(b_large_values, a_lot, large_values_out_p);
				int success = true;
				for (int i = 0; i < a_few; ++i)
					success &= (small_keys_out_p[i] == small_values_out_p[i] && small_values_out_p[i] == i + 1);
				Require(success == true);
				for (int i = 0; i < a_lot; ++i)
					success &= (large_keys_out_p[i] == large_values_out_p[i] && large_values_out_p[i] == i + 1);
				Require(success == true);
			}
		}
	}
}
Scenario("Parallel Radix Sort (Big Unsigneds)", "[sort][integration]") {
	Given("An arbitrary set of Big Unsigneds") {
		vector<BigUnsigned> small_input(a_few);
		vector<BigUnsigned> large_input(a_lot);

		for (int i = 0; i < a_few; ++i) initLongLongBU(&small_input[i], a_few - i);
		for (int i = 0; i < a_lot; ++i) initLongLongBU(&large_input[i], a_lot - i);

		When("these Big Unsigneds are sorted in parallel") {
			cl_int error = 0;
			cl::Buffer b_small_input, b_large_input;
			error |= CLFW::get(b_small_input, "b_small_input", a_few * sizeof(BigUnsigned));
			error |= CLFW::get(b_large_input, "b_large_input", a_lot * sizeof(BigUnsigned));
			error |= CLFW::Upload<BigUnsigned>(small_input, b_small_input);
			error |= CLFW::Upload<BigUnsigned>(large_input, b_large_input);
			error |= RadixSortBigUnsigned_p(b_small_input, a_few, 20, "a");
			error |= RadixSortBigUnsigned_p(b_large_input, a_lot, 20, "b");
			Require(error == CL_SUCCESS);
			Then("The Big Unsigneds are ordered assending") {
				vector<BigUnsigned> small_output_p(a_few), large_output_p(a_lot);
				error |= CLFW::Download<BigUnsigned>(b_small_input, a_few, small_output_p);
				error |= CLFW::Download<BigUnsigned>(b_large_input, a_lot, large_output_p);
				int success = true;
				for (int i = 0; i < a_few; ++i) {
					BigUnsigned temp;
					initLongLongBU(&temp, i + 1);
					success &= (compareBU(&small_output_p[i], &temp) == 0);
				}
				Require(success == true);
				for (int i = 0; i < a_lot; ++i) {
					BigUnsigned temp;
					initLongLongBU(&temp, i + 1);
					success &= (compareBU(&large_output_p[i], &temp) == 0);
				}
				Require(success == true);
			}
		}
	}
}
Scenario("Parallel Radix Sort (BU-Int Pairs by Key)", "[sort][integration]") {
	Given("An arbitrary set of unsigned key and integer value pairs") {
		vector<BigUnsigned> small_keys_in(a_few);
		vector<cl_int> small_values_in(a_few);
		vector<BigUnsigned> large_keys_in(a_lot);
		vector<cl_int> large_values_in(a_lot);

		for (int i = 0; i < a_few; ++i) { 
			initLongLongBU(&small_keys_in[i], a_few - i);
			small_values_in[i] = a_few - i; 
		}
		for (int i = 0; i < a_lot; ++i) { 
			initLongLongBU(&large_keys_in[i], a_lot - i);
			large_values_in[i] = a_lot - i;
		}

		When("these pairs are sorted by key in parallel") {
			cl_int error = 0;
			cl::Buffer b_small_keys, b_small_values, b_large_keys, b_large_values;
			error |= CLFW::get(b_small_keys, "b_small_keys", a_few * sizeof(BigUnsigned));
			error |= CLFW::get(b_small_values, "b_small_values", a_few * sizeof(cl_int));
			error |= CLFW::get(b_large_keys, "b_large_keys", a_lot * sizeof(BigUnsigned));
			error |= CLFW::get(b_large_values, "b_large_values", a_lot * sizeof(cl_int));
			error |= CLFW::Upload<BigUnsigned>(small_keys_in, b_small_keys);
			error |= CLFW::Upload<cl_int>(small_values_in, b_small_values);
			error |= CLFW::Upload<BigUnsigned>(large_keys_in, b_large_keys);
			error |= CLFW::Upload<cl_int>(large_values_in, b_large_values);
			Require(error == CL_SUCCESS);
			error |= RadixSortBUIntPairsByKey(b_small_keys, b_small_values, 20, a_few);
			Require(error == CL_SUCCESS);
			error |= RadixSortBUIntPairsByKey(b_large_keys, b_large_values, 20, a_lot);
			Require(error == CL_SUCCESS);
			Then("The key value pairs are ordered by keys assending") {
				vector<BigUnsigned> small_keys_out_p(a_few), large_keys_out_p(a_lot);
				vector<cl_int> large_values_out_p(a_lot), small_values_out_p(a_few);
				error |= CLFW::Download<BigUnsigned>(b_small_keys, a_few, small_keys_out_p);
				error |= CLFW::Download<BigUnsigned>(b_large_keys, a_lot, large_keys_out_p);
				error |= CLFW::Download<cl_int>(b_small_values, a_few, small_values_out_p);
				error |= CLFW::Download<cl_int>(b_large_values, a_lot, large_values_out_p);
				int success = true;
				for (int i = 0; i < a_few; ++i) {
					success &= (small_values_out_p[i] && small_values_out_p[i] == i + 1);
					BigUnsigned temp;
					initLongLongBU(&temp, i + 1);
					success &= (compareBU(&small_keys_out_p[i], &temp)==0);
				}
				Require(success == true);
				for (int i = 0; i < a_lot; ++i) {
					success &= (large_values_out_p[i] && large_values_out_p[i] == i + 1);
					BigUnsigned temp;
					initLongLongBU(&temp, i + 1);
					success &= (compareBU(&large_keys_out_p[i], &temp)==0);
				}
				Require(success == true);
			}
		}
	}
}

/* Z-Order Kernels*/
Scenario("Quantize Points", "[zorder]") {
	Given("A bounding box and a resolution") {
		floatn minimum = make_floatn(0.0, 0.0);
		floatn maximum = make_floatn(1000.0, 1000.0);
		BoundingBox bb = BB_initialize(&minimum, &maximum);
		int resolution_width = 64;
		Given("an arbitrary float2, say a bb min, bb max, and a point in the middle") {
			floatn middle = make_float2(500.0, 500.0);
			Then("those points can be quantized, or mapped, to the cooresponding quantized integer.") {
				intn quantized_min = QuantizePoint(&minimum, &minimum, resolution_width, bb.maxwidth);
				intn quantized_max = QuantizePoint(&maximum, &minimum, resolution_width, bb.maxwidth);
				intn quantized_middle = QuantizePoint(&middle, &minimum, resolution_width, bb.maxwidth);
				Require(quantized_min.x == 0);
				Require(quantized_min.y == 0);
				Require(quantized_max.x == resolution_width - 1);
				Require(quantized_max.x == resolution_width - 1);
				Require(quantized_middle.x == resolution_width / 2);
				Require(quantized_middle.y == resolution_width / 2);
			}
		}
		Given("an arbitrary set of float2s within the bounding box") {
			vector<float2> small_input = generateDeterministicRandomFloat2s(a_few, 0, 0.0, 1000.0);
			vector<float2> large_input = generateDeterministicRandomFloat2s(a_lot, 0, 0.0, 1000.0);
			When("these points are quantized in series") {
				vector<int2> small_output_s(a_few);
				vector<int2> large_output_s(a_lot);
				QuantizePoints_s(small_input, bb, resolution_width, small_output_s);
				QuantizePoints_s(large_input, bb, resolution_width, large_output_s);
				Then("the series results match the parallel results") {
					cl_int error = 0;
					cl::Buffer b_small_input, b_small_output, b_large_input, b_large_output;
					vector<intn> small_output_p(a_few), large_output_p(a_lot);
					error |= CLFW::get(b_small_input, "b_small_input", a_few * sizeof(int2));
					error |= CLFW::get(b_large_input, "b_large_input", a_lot * sizeof(int2));
					error |= CLFW::Upload<floatn>(small_input, b_small_input);
					error |= CLFW::Upload<floatn>(large_input, b_large_input);
					error |= QuantizePoints_p(b_small_input, a_few, bb, resolution_width, "a", b_small_output);
					error |= QuantizePoints_p(b_large_input, a_lot, bb, resolution_width, "b", b_large_output);
					error |= CLFW::Download<intn>(b_small_output, a_few, small_output_p);
					error |= CLFW::Download<intn>(b_large_output, a_lot, large_output_p);
					Require(error == CL_SUCCESS);
					int success = true;
          for (int i = 0; i < a_few; ++i) {
            success &= (small_output_s[i] == small_output_p[i]);
            if (!success) {
              success &= (large_output_s[i] == large_output_p[i]);
            }
          }
          for (int i = 0; i < a_lot; ++i) {
            success &= (large_output_s[i] == large_output_p[i]);
            if (!success) {
              success &= (large_output_s[i] == large_output_p[i]);
              cout<<large_output_s[i] << " vs " << large_output_p[i]<<endl;
            }
          }
					Require(success == true);
				}
			}
		}
	}
}
Scenario("QPoints to ZPoints", "[zorder]") {
	Given("the number of bits supported by the z-ordering") {
		Given("an arbitrary set of positive int2s") {
			vector<intn> small_input = generateDeterministicRandomInt2s(a_few, 1, 0, 1024);
			vector<intn> large_input = generateDeterministicRandomInt2s(a_lot, 2, 0, 1024);
			vector<BigUnsigned> small_output_s(a_few), large_output_s(a_lot);
			When("these points are placed on a Z-Order curve in series") {
				QPointsToZPoints_s(small_input, 30, small_output_s);
				QPointsToZPoints_s(large_input, 30, large_output_s);
				Then("the series results match the parallel results") {
					cl_int error = 0;
					vector<BigUnsigned> small_output_p(a_few), large_output_p(a_lot);
					cl::Buffer b_small_input, b_small_output, b_large_input, b_large_output;
					error |= CLFW::get(b_small_input, "b_small_input", a_few * sizeof(BigUnsigned));
					error |= CLFW::get(b_large_input, "b_large_input", a_lot * sizeof(BigUnsigned));
					error |= CLFW::Upload<intn>(small_input, b_small_input);
					error |= CLFW::Upload<intn>(large_input, b_large_input);
					error |= QPointsToZPoints_p(b_small_input, a_few, 30, "a", b_small_output);
					error |= QPointsToZPoints_p(b_large_input, a_lot, 30, "b", b_large_output);
					error |= CLFW::Download<BigUnsigned>(b_small_output, a_few, small_output_p);
					error |= CLFW::Download<BigUnsigned>(b_large_output, a_lot, large_output_p);
					Require(error == CL_SUCCESS);
					int success = true;
					for (int i = 0; i < a_few; ++i) success &= (compareBU(&small_output_s[i], &small_output_p[i]) == 0);
					for (int i = 0; i < a_lot; ++i) success &= (compareBU(&large_output_s[i], &large_output_p[i]) == 0);
					Require(success == true);
				}
			}
		}
	}
}

/* Unique Kernels */
Scenario("Unique Sorted BigUnsigned", "[sort][unique]") {
	Given("An ascending sorted set of BigUnsigneds") {
		TODO("delete large binary for this");
		vector<BigUnsigned> small_zpoints = readFromFile<BigUnsigned>("TestData//few_non-unique_s_zpoints.bin", a_few);
		vector<BigUnsigned> large_zpoints = readFromFile<BigUnsigned>("TestData//lot_non-unique_s_zpoints.bin", a_lot);
		When("those BigUnsigneds are uniqued in parallel") {
			cl_int error = 0, newSmallSize, newLargeSize;
			cl::Buffer b_small_zpoints, b_unique_small_zpoints, b_large_zpoints, b_unique_large_zpoints;
			error |= CLFW::get(b_small_zpoints, "b_small_zpoints", a_few * sizeof(BigUnsigned));
			error |= CLFW::get(b_large_zpoints, "b_large_zpoints", a_lot * sizeof(BigUnsigned));
			error |= CLFW::Upload<BigUnsigned>(small_zpoints, b_small_zpoints);
			error |= CLFW::Upload<BigUnsigned>(large_zpoints, b_large_zpoints);
			error |= UniqueSorted(b_small_zpoints, a_few, "a", newSmallSize);
			error |= UniqueSorted(b_large_zpoints, a_lot, "b", newLargeSize);
			vector<BigUnsigned> p_small_zpoints(newSmallSize);
			vector<BigUnsigned> p_large_zpoints(newLargeSize);
			error |= CLFW::Download<BigUnsigned>(b_small_zpoints, newSmallSize, p_small_zpoints);
			error |= CLFW::Download<BigUnsigned>(b_large_zpoints, newLargeSize, p_large_zpoints);
			Then("the resulting set should match the uniqued series set.") {
				auto sm_last = unique(small_zpoints.begin(), small_zpoints.end(), weakEqualsBU);
				auto lg_last = unique(large_zpoints.begin(), large_zpoints.end(), weakEqualsBU);
				small_zpoints.erase(sm_last, small_zpoints.end());
				large_zpoints.erase(lg_last, large_zpoints.end());
				Require(small_zpoints.size() == newSmallSize);
				Require(large_zpoints.size() == newLargeSize);
				cl_int success = 1;
				for (int i = 0; i < small_zpoints.size(); ++i)
					success &= (compareBU(&small_zpoints[i], &p_small_zpoints[i]) == 0);
				for (int i = 0; i < large_zpoints.size(); ++i)
					success &= (compareBU(&large_zpoints[i], &p_large_zpoints[i]) == 0);
				Require(success == true);
			}
		}
	}
}
Scenario("Unique Sorted BigUnsigned color pairs", "[sort][unique]") {
	Given("An ascending sorted set of BigUnsigneds") {
		vector<BigUnsigned> small_keys(a_few);
		vector<BigUnsigned> large_keys(a_lot);
		vector<cl_int> small_values(a_few);
		vector<cl_int> large_values(a_lot);

		for (int i = 0; i < a_few; ++i) {
			small_values[i] = i / 2;
			initLongLongBU(&small_keys[i], i / 2);
		}
		for (int i = 0; i < a_lot; ++i) {
			large_values[i] = i / 2;
			initLongLongBU(&large_keys[i], i / 2);
		}

		When("those BigUnsigneds are uniqued in parallel") {
			cl_int error = 0, newSmallSize, newLargeSize;
			cl::Buffer b_small_keys, b_unique_small_keys, b_large_keys, b_unique_large_keys;
			cl::Buffer b_small_values, b_unique_small_values, b_large_values, b_unique_large_values;
			error |= CLFW::get(b_small_keys, "b_small_zpoints", a_few * sizeof(BigUnsigned));
			error |= CLFW::get(b_large_keys, "b_large_zpoints", a_lot * sizeof(BigUnsigned));
			error |= CLFW::get(b_small_values, "b_small_values", a_few * sizeof(cl_int));
			error |= CLFW::get(b_large_values, "b_large_values", a_lot * sizeof(cl_int));
			error |= CLFW::Upload<BigUnsigned>(small_keys, b_small_keys);
			error |= CLFW::Upload<BigUnsigned>(large_keys, b_large_keys);
			error |= CLFW::Upload<cl_int>(small_values, b_small_values);
			error |= CLFW::Upload<cl_int>(large_values, b_large_values);
			error |= UniqueSortedBUIntPair(b_small_keys, b_small_values, a_few, "a", newSmallSize);
			error |= UniqueSortedBUIntPair(b_large_keys, b_large_values, a_lot, "b", newLargeSize);
			vector<BigUnsigned> p_small_zpoints(newSmallSize);
			vector<BigUnsigned> p_large_zpoints(newLargeSize);
			error |= CLFW::Download<BigUnsigned>(b_small_keys, newSmallSize, p_small_zpoints);
			error |= CLFW::Download<BigUnsigned>(b_large_keys, newLargeSize, p_large_zpoints);
			Then("the resulting set should match the uniqued series set.") {
				auto sm_last = unique(small_keys.begin(), small_keys.end(), weakEqualsBU);
				auto lg_last = unique(large_keys.begin(), large_keys.end(), weakEqualsBU);
				small_keys.erase(sm_last, small_keys.end());
				large_keys.erase(lg_last, large_keys.end());
				Require(small_keys.size() == newSmallSize);
				Require(large_keys.size() == newLargeSize);
				cl_int success = 1;
				for (int i = 0; i < small_keys.size(); ++i)
					success &= (compareBU(&small_keys[i], &p_small_zpoints[i]) == 0);
				for (int i = 0; i < large_keys.size(); ++i)
					success &= (compareBU(&large_keys[i], &p_large_zpoints[i]) == 0);
				Require(success == true);
			}
		}
	}
}

/* Tree Building Kernels */
Scenario("Build Binary Radix Tree", "[selected][tree]") {
	Given("A set of unique ordered zpoints") {
		int lotmbits = 48;
		int fewmbits = 6;
		vector<BigUnsigned> small_zpoints = readFromFile<BigUnsigned>("TestData//few_u_s_zpoints.bin", a_few);
		vector<BigUnsigned> large_zpoints = readFromFile<BigUnsigned>("TestData//lot_u_s_zpoints.bin", a_lot);
		When("we build a binary radix tree using these points in parallel") {
			cl_int error = 0;
			vector<BrtNode> small_brt_p(a_few);
			vector<BrtNode> large_brt_p(a_lot);
			cl::Buffer b_small_zpoints, b_large_zpoints, b_small_brt, b_large_brt;
			error |= CLFW::get(b_small_zpoints, "b_small_zpoints", a_few * sizeof(BigUnsigned));
			error |= CLFW::get(b_large_zpoints, "b_large_zpoints", a_lot * sizeof(BigUnsigned));
			error |= CLFW::Upload<BigUnsigned>(small_zpoints, b_small_zpoints);
			error |= CLFW::Upload<BigUnsigned>(large_zpoints, b_large_zpoints);
			error |= BuildBinaryRadixTree_p(b_small_zpoints, a_few, fewmbits, "a", b_small_brt);
			error |= BuildBinaryRadixTree_p(b_large_zpoints, a_lot, lotmbits, "b", b_large_brt);
			error |= CLFW::Download<BrtNode>(b_small_brt, a_few, small_brt_p);
			//writeToFile<BrtNode>(small_brt_p, "TestData//few_brt.bin");
			error |= CLFW::Download<BrtNode>(b_large_brt, a_lot, large_brt_p);
			Require(error == CL_SUCCESS);
			Then("the resulting binary radix tree should be valid") {
				/* Precomputed */
				vector<BrtNode> small_brt_s = readFromFile<BrtNode>("TestData//few_brt.bin", a_few);
				vector<BrtNode> large_brt_s = readFromFile<BrtNode>("TestData//lot_brt.bin", a_lot);
				int success = true;
				for (int i = 0; i < a_few; ++i) {
					success &= (true == compareBrtNode(&small_brt_p[i], &small_brt_s[i]));
					if (!success)
						success &= (true == compareBrtNode(&small_brt_p[i], &small_brt_s[i]));
				}
				for (int i = 0; i < a_lot; ++i) {
					success &= (true == compareBrtNode(&large_brt_p[i], &large_brt_s[i]));
				}
				Require(success == true);
			}
		}
	}
}
Scenario("Build Colored Binary Radix Tree", "[selected][tree]") {
	Given("A set of colored unique ordered zpoints") {
		cl_int mbits = readFromFile<cl_int>("TestData//simple//mbits.bin");
		cl_int totalPoints = readFromFile<cl_int>("TestData//simple//uniqueTotalPoints.bin");
		vector<BigUnsigned> zpoints = readFromFile<BigUnsigned>("TestData//simple//uniqueZPoints.bin", totalPoints);
		vector<cl_int> leafColors = readFromFile<cl_int>("TestData//simple//uniqueColors.bin", totalPoints);
		When("we build a colored binary radix tree using these points in series") {
			vector<BrtNode> brt_s;
			vector<cl_int> brtColors_s;
			BuildColoredBinaryRadixTree_s(zpoints, leafColors, mbits, brt_s, brtColors_s);
			clearScreen();
			DrawBRT(brt_s, brtColors_s);
			refresh();
			Then("the resulting binary radix tree and cooresponding colors should be valid") {
				vector<BrtNode> brt_f = readFromFile<BrtNode>("TestData//simple//brt.bin", totalPoints - 1);
				vector<cl_int> brtColors_f = readFromFile<cl_int>("TestData//simple//unpropagatedBrtColors.bin", totalPoints - 1);
				cl_int success = true;
				for (int i = 0; i < brt_s.size(); ++i) {
					success &= (brtColors_s[i] == brtColors_f[i]);
					success &= (compareBrtNode(&brt_s[i], &brt_f[i]));

					if (brt_s[i].parent != -1) {
						BrtNode p = brt_s[brt_s[i].parent];
						assert(p.left == i || p.left + 1 == i);
					}
				}
				Require(success == true);
			}
			Then("the series results should match the parallel results") {
				cl_int error = 0;
				vector<BrtNode> brt_p;
				vector<cl_int> brtColors_p;
				cl::Buffer b_zpoints, b_leafColors, b_brt, b_brtColors;
				error |= CLFW::get(b_zpoints, "b_zpoints", totalPoints * sizeof(BigUnsigned));
				error |= CLFW::get(b_leafColors, "b_leafColors", totalPoints * sizeof(cl_int));
				error |= CLFW::Upload<BigUnsigned>(zpoints, b_zpoints);
				error |= CLFW::Upload<cl_int>(leafColors, b_leafColors);
				error |= BuildColoredBinaryRadixTree_p(b_zpoints, b_leafColors, totalPoints, mbits, "", b_brt, b_brtColors);
				error |= CLFW::Download<BrtNode>(b_brt, totalPoints - 1, brt_p);
				error |= CLFW::Download<cl_int>(b_brtColors, totalPoints - 1, brtColors_p);
				Require(error == CL_SUCCESS);
				cl_int success = true;
				for (int i = 0; i < brt_s.size(); ++i) {
					success &= (brtColors_s[i] == brtColors_p[i]);
					success &= (compareBrtNode(&brt_s[i], &brt_p[i]));
				}
				Require(success == true);
			}
		}
	}
}

{
	Given("a colored binary radix tree") {
		cl_int totalPoints = readFromFile<cl_int>("TestData//simple//numPoints.bin");
		vector<BrtNode> brt = readFromFile<BrtNode>("TestData//simple//brt.bin", totalPoints - 1);
		vector<cl_int> brtColors_s = readFromFile<cl_int>("TestData//simple//unpropagatedBrtColors.bin", totalPoints - 1);
		vector<cl_int> brtColors_f = readFromFile<cl_int>("TestData//simple//unpropagatedBrtColors.bin", totalPoints - 1);

		When("we propagate the BRT colors up the tree in series") {
			TODO("create brt leaf mapping");
			PropagateBRTColors_s(brt, brtColors_s);
			clearScreen();
			DrawBRT(brt, brtColors_s);
			refresh();
			Then("the results should be valid") {
				TODO("test this");
			}
			Then("the series results should match the parallel results") {
				cl_int error = 0;
				cl::Buffer b_brt, b_brtColors;
				vector<cl_int> brtColors_p;
				error |= CLFW::get(b_brt, "brt", (totalPoints - 1) * sizeof(BrtNode));
				error |= CLFW::get(b_brtColors, "b_brtColors", (totalPoints - 1) * sizeof(cl_int));
				error |= CLFW::Upload<BrtNode>(brt, b_brt);
				error |= CLFW::Upload<cl_int>(brtColors_f, b_brtColors);
				error |= PropagateBRTColors_p(b_brt, b_brtColors, totalPoints - 1, "");
				error |= CLFW::Download<cl_int>(b_brtColors, totalPoints - 1, brtColors_p);

				Require(error == CL_SUCCESS);

				cl_int success = true;
				for (cl_int i = 0; i < totalPoints - 1; i++) {
					success &= brtColors_p[i] == brtColors_s[i];
				}
				Require(success == true);
			}
		}
	}
}
Scenario("Build Quadtree", "[tree]") {
	Given("a binary radix tree") {
		auto small_brt = readFromFile<BrtNode>("TestData//few_brt.bin", a_few);
		auto large_brt = readFromFile<BrtNode>("TestData//lot_brt.bin", a_lot);
		When("we use that binary radix tree to build an octree in parallel") {
			cl_int error = 0, small_octree_size, large_octree_size;
			cl::Buffer b_small_brt, b_large_brt, b_small_octree, b_large_octree, nullBuffer;
			error |= CLFW::get(b_small_brt, "b_small_brt", a_few * sizeof(BrtNode));
			error |= CLFW::get(b_large_brt, "b_large_brt", a_lot * sizeof(BrtNode));
			error |= CLFW::Upload<BrtNode>(small_brt, b_small_brt);
			error |= CLFW::Upload<BrtNode>(large_brt, b_large_brt);
			error |= BinaryRadixToOctree_p(b_small_brt, false, nullBuffer, a_few, "a", b_small_octree, small_octree_size);
			vector<OctNode> small_octree_p(small_octree_size);// , 
			error |= CLFW::Download<OctNode>(b_small_octree, small_octree_size, small_octree_p);
			error |= BinaryRadixToOctree_p(b_large_brt, false, nullBuffer, a_lot, "b", b_large_octree, large_octree_size);
			vector<OctNode> large_octree_p(large_octree_size);
			error |= CLFW::Download<OctNode>(b_large_octree, large_octree_size, large_octree_p);

			Then("our results should be valid") {
				Require(small_octree_size == 11);
				Require(large_octree_size == 333351);
				
				auto small_octree_s = readFromFile<OctNode>("TestData//few_octree.bin", 11);
				auto large_octree_s = readFromFile<OctNode>("TestData//lot_octree.bin", 333351);

				int success = true;
				for (int i = 0; i < 11; ++i) {
					success &= (compareOctNode(&small_octree_s[i], &small_octree_p[i]));
				}
				for (int i = 0; i < 333351; ++i) {
					success &= (compareOctNode(&large_octree_s[i], &large_octree_p[i]));
					if (!success) {
						WARN("ERROR: octnode " << i << " does not match!");
						break;
					}
				}
				Require(success == true);
			}
		}
	}
}
Scenario("GenerateLeaves", "[tree]") {
	Given("an octnode") {
		OctNode n;
		n.children[0] = -1;
		n.children[1] = 1;
		n.children[2] = -1;
		n.children[3] = 0;
		When("we generate the leaves for that octnode") {
			vector<Leaf> leaves(4);
			vector<cl_int> predicates(4);
			for (int i = 0; i < 4; ++i)
				ComputeLeaves(&n, leaves.data(), predicates.data(), 1, i);
			Then("the results should be valid") {
				Require(leaves[0].parent == leaves[2].parent);
				Require(leaves[2].parent == 0);
				Require(leaves[0].quadrant == 0);
				Require(leaves[2].quadrant == 2);
				Require(predicates[0] == 1);
				Require(predicates[1] == 0);
				Require(predicates[2] == 1);
				Require(predicates[3] == 0);
			}
		}
	}
	Given("an octree") {
		auto small_octree = readFromFile<OctNode>("TestData//few_octree.bin", 11);
		auto large_octree = readFromFile<OctNode>("TestData//lot_octree.bin", 333351);

		When("we generate the leaves of this octree in series") {
			vector<cl_int> small_pred_s(4 * 11), large_pred_s(4 * 333351);
			vector<Leaf> small_leaves_s(4 * 11), large_leaves_s(4 * 333351);
			GenerateLeaves_s(small_octree, 11, small_leaves_s, small_pred_s);
			GenerateLeaves_s(large_octree, 333351, large_leaves_s, large_pred_s);
			Then("the parallel results match the serial ones") {
				cl_int error = 0;
				vector<cl_int> small_pred_p(4*11), large_pred_p(4*333351);
				vector<Leaf> small_leaves_p(4*11), large_leaves_p(4*333351);
				cl::Buffer b_small_octree, b_large_octree, b_small_pred, b_large_pred, b_small_leaves, b_large_leaves;
				error |= CLFW::get(b_small_octree, "b_small_octree", 4 * 11 * sizeof(OctNode));
				error |= CLFW::get(b_large_octree, "b_large_octree", 4 * 333351 * sizeof(OctNode));
				error |= CLFW::Upload<OctNode>(small_octree, b_small_octree);
				error |= CLFW::Upload<OctNode>(large_octree, b_large_octree);
				error |= GenerateLeaves_p(b_small_octree, 11, b_small_leaves, b_small_pred);
				error |= GenerateLeaves_p(b_large_octree, 333351, b_large_leaves, b_large_pred);
				error |= CLFW::Download(b_small_pred, 4 * 11, small_pred_p);
				error |= CLFW::Download(b_large_pred, 4 * 333351, large_pred_p);
				error |= CLFW::Download(b_small_leaves, 4 * 11, small_leaves_p);
				error |= CLFW::Download(b_large_leaves, 4 * 333351, large_leaves_p);
				Require(error == CL_SUCCESS);
				cl_int success = true;
				for (int i = 0; i < 4 * 11; ++i) {
					success &= (compareLeaf(&small_leaves_s[i], &small_leaves_p[i]));
					success &= (small_pred_s[i] == small_pred_p[i]);
				}
				for (int i = 0; i < 4 * 333351; ++i) {
					success &= (compareLeaf(&large_leaves_s[i], &large_leaves_p[i]));
					success &= (large_pred_s[i] == large_pred_p[i]);
				}
				Require(success == true);
			}
		}
	}
}

/* Ambiguous cell detection kernels */
Scenario("Get LCPs From Lines", "[conflict]") {
	Given("two z-order points and a line connecting them") {
		int mbits = 8;
		vector<BigUnsigned> p(2);
		initBlkBU(&p[0], 240); //11110000
		initBlkBU(&p[1], 243); //11110011
		Line l;
		l.first = 0; l.second = 1;
		When("we get the LCP from this line's points") {
			LCP lcp = {};
			GetLCPFromLine(&l, p.data(), &lcp, mbits, 0);
			Require(lcp.len == 6);
			Require(lcp.bu.blk[0] == 60);
		}
	}
	Given("N z-ordered points and lines, and the number of bits per zpoint") {
		int mbits = 20;
		vector<BigUnsigned> small_zpoints = readFromFile<BigUnsigned>("TestData//few_u_s_zpoints.bin", a_few);
		vector<BigUnsigned> large_zpoints = readFromFile<BigUnsigned>("TestData//lot_u_s_zpoints.bin", a_lot);
		vector<Line> small_lines(a_few), large_lines(a_lot);
		for (int i = 0; i < a_few; ++i) {
			Line l;
			l.first = i;
			l.second = (i == a_few - 1) ? 0 : (i + 1);
			small_lines[i] = l;
		}
		for (int i = 0; i < a_lot; ++i) {
			Line l;
			l.first = i;
			l.second = (i == a_lot - 1) ? 0 : (i + 1);
			large_lines[i] = l;
		}

		When("we compute the lcps for these lines in series") {
			vector<LCP> small_lcps_s(a_few), large_lcps_s(a_lot);
			GetLineLCP_s(small_lines, small_zpoints, mbits, small_lcps_s);
			GetLineLCP_s(large_lines, large_zpoints, mbits, large_lcps_s);
			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<LCP> small_lcps_p(a_few), large_lcps_p(a_lot);
				cl::Buffer b_small_lines, b_large_lines, b_small_zpoints, b_large_zpoints, b_small_lcps, b_large_lcps;
				error |= CLFW::get(b_small_lines, "b_small_lines", a_few * sizeof(Line));
				error |= CLFW::get(b_large_lines, "b_large_lines", a_lot * sizeof(Line));
				error |= CLFW::get(b_small_zpoints, "b_small_zpoints", a_few * sizeof(BigUnsigned));
				error |= CLFW::get(b_large_zpoints, "b_large_zpoints", a_lot * sizeof(BigUnsigned));
				error |= CLFW::Upload<Line>(small_lines, b_small_lines);
				error |= CLFW::Upload<Line>(large_lines, b_large_lines);
				error |= CLFW::Upload<BigUnsigned>(small_zpoints, b_small_zpoints);
				error |= CLFW::Upload<BigUnsigned>(large_zpoints, b_large_zpoints);
				error |= GetLineLCPs_p(b_small_lines, a_few, b_small_zpoints, mbits, b_small_lcps);
				error |= GetLineLCPs_p(b_large_lines, a_lot, b_large_zpoints, mbits, b_large_lcps);
				error |= CLFW::Download<LCP>(b_small_lcps, a_few, small_lcps_p);
				error |= CLFW::Download<LCP>(b_large_lcps, a_lot, large_lcps_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i) {
					success &= (compareLCP(&small_lcps_s[i], &small_lcps_p[i]) == 0);
					if (!success)
						success &= (compareLCP(&small_lcps_s[i], &small_lcps_p[i]) == 0);
				}
				for (int i = 0; i < a_lot; ++i)
				{
					success &= (compareLCP(&large_lcps_s[i], &large_lcps_p[i]) == 0);
					if (!success) 
						success &= (compareLCP(&large_lcps_s[i], &large_lcps_p[i]) == 0);
				}
				Require(success == true);
			}
		}
	}
}
Scenario("Look Up Octnode From LCP", "[conflict]") {
	Given("an octree and a LCP") {
		cl_int numOctNodes = readFromFile<cl_int>("TestData//simple//numOctNodes.bin");
		vector<OctNode> octree = readFromFile<OctNode>("TestData//simple//octree.bin", numOctNodes);
		LCP testLCP;
		testLCP.bu.len = 1;
		testLCP.bu.blk[0] = 15;
		testLCP.len = 4;
		Then("we can find the bounding octnode for that LCP") {
			int index = getOctNode(testLCP.bu, testLCP.len, octree.data(), &octree[0]);
			Require(index == 17);
		}
	}
	Given("An octree, and the LCPs of the facets generating that octree") {
		cl_int numOctNodes = readFromFile<cl_int>("TestData//simple//numOctNodes.bin");
		cl_int numLines = readFromFile<cl_int>("TestData//simple//numLines.bin");
		vector<OctNode> octree = readFromFile<OctNode>("TestData//simple//octree.bin", numOctNodes);
		vector<LCP> lineLCPs = readFromFile<LCP>("TestData//simple//line_lcps.bin", numLines);
		When("we look up the containing octnode in series") {
			// Precomputed
			vector<cl_int> s_LCPToOctnode = readFromFile<cl_int>("TestData//simple//LCPToOctNode.bin", numLines);
			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<cl_int> p_LCPToOctnode(numLines);
				cl::Buffer b_octree, b_lineLCPs, b_LCPToOctnode;
				error |= CLFW::get(b_octree, "b_octree", numOctNodes * sizeof(OctNode));
				error |= CLFW::get(b_lineLCPs, "b_lineLCPs", nextPow2(numOctNodes * sizeof(LCP)));
				error |= CLFW::Upload(octree, b_octree);
				error |= CLFW::Upload(lineLCPs, b_lineLCPs);
				error |= LookUpOctnodeFromLCP_p(b_lineLCPs, numLines, b_octree, b_LCPToOctnode);
				error |= CLFW::Download(b_LCPToOctnode, numLines, p_LCPToOctnode);
				Require(error == CL_SUCCESS);
				cl_int success = 1;
				for (int i = 0; i < numLines; ++i)
					success &= (p_LCPToOctnode[i] == s_LCPToOctnode[i]);
				Require(success == true);
			}
		}
	}
}
Scenario("Get Octnode LCP Bounds", "[conflict]") {
	Given("a LCP to Octnode mapping, ordered by octnode ascending") {
		//Note: the LCP to octnode mapping ordered by octnode ascending can be aquired using the RadixSortPairsByKey,
		cl_int numLines = readFromFile<cl_int>("TestData//simple//numLines.bin");
		cl_int numOctNodes = readFromFile<cl_int>("TestData//simple//numOctNodes.bin");
		vector<cl_int> LCPToOctNode = readFromFile<cl_int>("TestData//simple//Sorted_LCPToOctNode.bin", numLines);
		When("we get the octnode facet bounds in series") {
			vector<Pair> s_LCPBounds;
			GetLCPBounds_s(LCPToOctNode, numLines, numOctNodes, s_LCPBounds);
			Then("the series results are valid") {
				vector<Pair> f_LCPBounds = readFromFile<Pair>("TestData//simple//LCPBounds.bin", numOctNodes);
				cl_int success = true;
				for (int i = 0; i < numOctNodes; i++) {
					success &= (s_LCPBounds[i].first == f_LCPBounds[i].first);
					success &= (s_LCPBounds[i].last == f_LCPBounds[i].last);
				}
				Require(success == true);
				And_then("the series results match the parallel results") {
					cl_int error = 0;
					vector<Pair> p_LCPBounds(numOctNodes);
					cl::Buffer b_LCPToOctNode, b_LCPBounds;
					error |= CLFW::get(b_LCPToOctNode, "b_LCPToOctNode", sizeof(cl_int) * numLines);
					error |= CLFW::Upload(LCPToOctNode, b_LCPToOctNode);
					error |= GetLCPBounds_p(b_LCPToOctNode, numLines, numOctNodes, b_LCPBounds);
					error |= CLFW::Download(b_LCPBounds, numOctNodes, p_LCPBounds);
					cl_int success = true;
					for (int i = 0; i < numOctNodes; i++) {
						success &= (s_LCPBounds[i].first == p_LCPBounds[i].first);
						success &= (s_LCPBounds[i].last == p_LCPBounds[i].last);
					}
					Require(success == true);
				}
			}
		}
	}
}
Scenario("Find Conflict Cells", "[conflict]") {
	Given("An octree, that octree's leaves, a bcell to \n"
		+ "line mapping (see paper), bcell index bounds for \n"
		+ "each internal octnode, the lines and cooresponding \n"
		+ "points used to generate the octree, and the octree width") 
	{
		cl_int f_numOctnodes							=		readFromFile<cl_int>("TestData//simple//numOctNodes.bin");
		cl_int f_numLeaves								=		readFromFile<cl_int>("TestData//simple//numLeaves.bin");
		cl_int f_numLines									=		readFromFile<cl_int>("TestData//simple//numLines.bin");
		cl_int f_numPoints								=		readFromFile<cl_int>("TestData//simple//numPoints.bin");
		cl_int f_qwidth										=		readFromFile<cl_int>("TestData//simple//qwidth.bin");
		vector<OctNode> f_octree					=		readFromFile<OctNode>("TestData//simple//octree.bin",					f_numOctnodes);
		vector<Leaf> f_leaves							=		readFromFile<Leaf>("TestData//simple//leaves.bin",						f_numLeaves);
		vector<cl_int> f_LCPToLine				=		readFromFile<cl_int>("TestData//simple//LCPToLine.bin",	f_numLines);
		vector<Pair> f_LCPBounds					=		readFromFile<Pair>("TestData//simple//LCPBounds.bin",				f_numOctnodes);
		vector<Line> f_lines							=		readFromFile<Line>("TestData//simple//lines.bin",							f_numLines);
		vector<intn> f_qpoints						=		readFromFile<intn>("TestData//simple//qpoints.bin",						f_numPoints);
		vector<Conflict> f_conflicts			=		readFromFile<Conflict>("TestData//simple//sparseConflicts.bin",			f_numLeaves);

		When("we use this data to find conflict cells in series") {
			vector<Conflict> s_conflicts;
			FindConflictCells_s(f_octree, f_leaves, f_LCPToLine, f_LCPBounds, f_lines, f_qpoints, f_qwidth, s_conflicts);
			Then("the results are valid") {
				cl_int success = true;
				for (int i = 0; i < f_numLeaves; ++i)
					success &= compareConflict(&s_conflicts[i], &f_conflicts[i]);
				Require(success == true);
			}
			Then("the series results match the parallel results") {
					cl_int error = 0;
					vector<Conflict> p_conflicts(f_numLeaves);
					cl::Buffer b_octree, b_leaves, b_LCPToLine, b_LCPBounds, b_lines, b_qpoints, b_conflicts;
					error |= CLFW::get(b_octree, "b_octree", sizeof(OctNode) * f_numOctnodes);
					error |= CLFW::get(b_leaves, "b_leaves", sizeof(Leaf) * f_numLeaves);
					error |= CLFW::get(b_LCPToLine, "b_LCPToLine", sizeof(cl_int) * f_numLines);
					error |= CLFW::get(b_LCPBounds, "b_LCPBounds", sizeof(Pair) * f_numOctnodes);
					error |= CLFW::get(b_lines, "b_lines", sizeof(Line) * f_numLines);
					error |= CLFW::get(b_qpoints, "b_qpoints", sizeof(intn) * f_numPoints);
					error |= CLFW::Upload(f_octree, b_octree);
					error |= CLFW::Upload(f_leaves, b_leaves);
					error |= CLFW::Upload(f_LCPToLine, b_LCPToLine);
					error |= CLFW::Upload(f_LCPBounds, b_LCPBounds);
					error |= CLFW::Upload(f_lines, b_lines);
					error |= CLFW::Upload(f_qpoints, b_qpoints);
					error |= FindConflictCells_p(b_octree, b_leaves, f_numLeaves, b_LCPToLine, b_LCPBounds, 
						b_lines, f_numLines, b_qpoints, f_qwidth, b_conflicts);
					error |= CLFW::Download(b_conflicts, f_numLeaves, p_conflicts);
					Require(error == 0);
					cl_int success = true;
					for (int i = 0; i < f_numLeaves; i++)
						success &= compareConflict(&p_conflicts[i], &s_conflicts[i]);
					Require(success == true);
				}
		}
	}
}

/* Ambiguous cell resolution kernels */
Scenario("Sample required resolution points", "[resolution]") {
	Given("a set of conflicts and the quantized points used to build the original octree") {
		cl_int numConflicts = readFromFile<cl_int>("TestData//simple//numConflicts.bin");
		cl_int numPoints = readFromFile<cl_int>("TestData//simple//numPoints.bin");
		vector<Conflict> conflicts = readFromFile<Conflict>("TestData//simple//conflicts.bin", numConflicts);
		vector<intn> qpoints = readFromFile<intn>("TestData//simple//qpoints.bin",  numPoints);
		
		When("we sample the information required to resolve these conflicts in series") {
			vector<ConflictInfo> conflictInfo_s;
			vector<cl_int> numPtsPerConflict_s;
			GetResolutionPointsInfo_s(conflicts, qpoints, conflictInfo_s, numPtsPerConflict_s);
			Then("our results are valid") {
				vector<ConflictInfo> conflictInfo_f = readFromFile<ConflictInfo>("TestData//simple//conflictInfo.bin", numConflicts);
				vector<cl_int>numPtsPerConflict_f = readFromFile<cl_int>("TestData//simple//numPtsPerConflict.bin", numConflicts);
				cl_int success = true;
				for (int i = 0; i < numConflicts; ++i) {
					success &= compareConflictInfo(&conflictInfo_s[i], &conflictInfo_f[i]);
					success &= (numPtsPerConflict_s[i] == numPtsPerConflict_f[i]);
					if (!success) {
						success &= compareConflictInfo(&conflictInfo_s[i], &conflictInfo_f[i]);
						success &= (numPtsPerConflict_s[i] == numPtsPerConflict_f[i]);
					}
				}
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<ConflictInfo> conflictInfo_p(numConflicts);
				vector<cl_int> numPtsPerConflict_p(numConflicts);
				cl::Buffer b_conflicts, b_qpoints, b_conflictInfo, b_numPtsPerConflict;
				error |= CLFW::get(b_conflicts, "b_conflicts", numConflicts * sizeof(Conflict));
				error |= CLFW::get(b_qpoints, "b_qpoints", numPoints * sizeof(intn));
				error |= CLFW::Upload<Conflict>(conflicts, b_conflicts);
				error |= CLFW::Upload<intn>(qpoints, b_qpoints);
				error |= GetResolutionPointsInfo_p(b_conflicts, numConflicts, b_qpoints, b_conflictInfo, b_numPtsPerConflict);
				error |= CLFW::Download<ConflictInfo>(b_conflictInfo, numConflicts, conflictInfo_p);
				error |= CLFW::Download<cl_int>(b_numPtsPerConflict, numConflicts, numPtsPerConflict_p);
				Require(error == CL_SUCCESS);
				cl_int success = true;
				for (int i = 0; i < numConflicts; ++i) {
					success &= compareConflictInfo(&conflictInfo_s[i], &conflictInfo_p[i]);
					success &= (numPtsPerConflict_s[i] == numPtsPerConflict_p[i]);
					if (!success) {
						success &= compareConflictInfo(&conflictInfo_s[i], &conflictInfo_p[i]);
						success &= (numPtsPerConflict_s[i] == numPtsPerConflict_p[i]);
						if (!success) {
							success &= compareConflictInfo(&conflictInfo_s[i], &conflictInfo_p[i]);
							success &= (numPtsPerConflict_s[i] == numPtsPerConflict_p[i]);
						}
					}
				}
				Require(success == true);
			}
		}
	}
}
Scenario("Predicate Conflict To Point", "[predication][resolution]") {
	Given("the scanned number of resolution points to create per conflict") {
		cl_int numConflicts = readFromFile<cl_int>("./TestData/simple/numConflicts.bin");
		cl_int numResPts = readFromFile<cl_int>("TestData//simple//numResPts.bin");
		vector<cl_int> scannedNumPtsPerConflict = readFromFile<cl_int>("TestData//simple//scannedNumPtsPerConflict.bin", numConflicts);
		When("we predicate the first point cooresponding to a conflict") {
			vector<cl_int> predicates(numResPts, 0);
			predPntToConflict(scannedNumPtsPerConflict.data(), predicates.data(), 0);
			predPntToConflict(scannedNumPtsPerConflict.data(), predicates.data(), numConflicts - 2);
			predPntToConflict(scannedNumPtsPerConflict.data(), predicates.data(), 10);
			Require(predicates[4] == 1);
			Require(predicates[56] == 1);
			Require(predicates[72] == 1);
		}
		When("we predicate the first points in series") {
			vector<cl_int> predication_s;
			PredicatePointToConflict_s(scannedNumPtsPerConflict, numResPts, predication_s);
			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<cl_int> predication_p;
				cl::Buffer b_scannedNumPtsPerConflict, b_predication;
				error |= CLFW::get(b_scannedNumPtsPerConflict, "b_scannedNumPtsPerConflict", numConflicts * sizeof(cl_int));
				error |= CLFW::Upload<cl_int>(scannedNumPtsPerConflict, b_scannedNumPtsPerConflict);
				error |= PredicatePointToConflict_p(b_scannedNumPtsPerConflict, numConflicts, numResPts, b_predication);
				error |= CLFW::Download<cl_int>(b_predication, numResPts, predication_p);
				Require(error == CL_SUCCESS);
				cl_int success = true;
				for (int i = 0; i < numConflicts; ++i) {
					success &= (predication_s[i] == predication_p[i]);
				}
				Require(success == true);
			}
		}
	}
}
Scenario("Get resolution points", "[resolution]") {
	Given("a set of conflicts and cooresponding conflict infos, a resolution point to conflict mapping, "
		+ "and the original quantized points used to build the octree") {
		cl_int numConflicts = readFromFile<cl_int>("TestData//simple//numConflicts.bin");
		cl_int numResPts = readFromFile<cl_int>("TestData//simple//numResPts.bin");
		cl_int numPts = readFromFile<cl_int>("TestData//simple//numPoints.bin");
		vector<Conflict> conflicts = readFromFile<Conflict>("TestData//simple//conflicts.bin", numConflicts);
		vector<ConflictInfo> conflictInfo = readFromFile<ConflictInfo>("TestData//simple//conflictInfo.bin", numConflicts);
		vector<cl_int> pntToConflict = readFromFile<cl_int>("TestData//simple//pntToConflict.bin", numResPts);
		vector<cl_int> scannedNumPtsPerConflict = readFromFile<cl_int>("TestData//simple//scannedNumPtsPerConflict.bin", numConflicts);
		vector<intn> qpoints = readFromFile<intn>("TestData//simple//qpoints.bin", numPts);
		vector<cl_int>numPtsPerConflict_f = readFromFile<cl_int>("TestData//simple//numPtsPerConflict.bin", numConflicts);
		vector<cl_int> test(numPtsPerConflict_f.size());
		When("we use this data to get the resolution points in series") {
			vector<intn> resPts_s;
			GetResolutionPoints_s(conflicts, conflictInfo, scannedNumPtsPerConflict, pntToConflict, numResPts, qpoints, resPts_s);
			Then("these points should be valid") {
				vector<intn> resPts_f = readFromFile<intn>("TestData//simple//resPts.bin", numResPts);
				cl_int success = true;
				for (int i = 0; i < numResPts; ++i) success &= (resPts_s[i] == resPts_f[i]);
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<intn> resPts_p;
				cl::Buffer b_conflicts, b_conflictInfo, b_scannedNumPtsPerConflict, b_pntToConflict, b_qpoints, b_resPts;
				error |= CLFW::get(b_conflicts, "b_conflicts", numConflicts * sizeof(Conflict));
				error |= CLFW::get(b_conflictInfo, "b_conflictInfo", numConflicts * sizeof(ConflictInfo));
				error |= CLFW::get(b_scannedNumPtsPerConflict, "b_scannedNumPtsPerConflict", numConflicts * sizeof(cl_int));
				error |= CLFW::get(b_pntToConflict, "b_pntToConflict", numResPts * sizeof(cl_int));
				error |= CLFW::get(b_qpoints, "b_qpoints", numPts * sizeof(intn));
				error |= CLFW::Upload<Conflict>(conflicts, b_conflicts);
				error |= CLFW::Upload<ConflictInfo>(conflictInfo, b_conflictInfo);
				error |= CLFW::Upload<cl_int>(scannedNumPtsPerConflict, b_scannedNumPtsPerConflict);
				error |= CLFW::Upload<cl_int>(pntToConflict, b_pntToConflict);
				error |= CLFW::Upload<intn>(qpoints, b_qpoints);
				error |= GetResolutionPoints_p(b_conflicts, b_conflictInfo, b_scannedNumPtsPerConflict, numResPts, b_pntToConflict, b_qpoints, b_resPts);
				error |= CLFW::Download<intn>(b_resPts, numResPts, resPts_p);
				Require(error == CL_SUCCESS);
				cl_int success = true;
				for (int i = 0; i < numResPts; ++i) success &= (resPts_s[i] == resPts_p[i]);
				Require(success == true);
			}
		}
	}
}
