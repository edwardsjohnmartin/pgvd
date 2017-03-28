/*
Unit Testing Notes:
- If a kernel is "embarrassingly parallel", then test with "aFew" elements
-- (Try optimizing test speed)

- If a kernel requires more intricate workgroup operations, (like reduction),
-- then test with both "aFew" and "aLot"
*/

#include "catch.hpp"
#include "Kernels/Kernels.h"
#include "HelperFunctions.hpp"
using namespace Kernels;
using namespace GLUtilities;

/* Reduction Kernels */
Scenario("Additive Reduction", "[reduction]") {
	Given("N random integers") {
		vector<cl_int>small_input = generateDeterministicRandomIntegers(a_few);
		vector<cl_int>large_input(a_lot, 1);
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
				cl::Buffer b_small_input, b_large_input, b_small_output, b_large_output;
				cl_int error = 0;
				error |= CLFW::getBuffer(b_small_input, "small", a_few * sizeof(cl_int));
				error |= CLFW::getBuffer(b_large_input, "large", a_lot * sizeof(cl_int));
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
//Scenario("Check Order", "[disabled][4way][sort][reduction]") {
//	Given("a list containing 2^n ordered big") {
//		TODO("run this check on n' elements, and then on 1 + n - n' elements");
//		vector<big> small_input(Kernels::nextPow2(a_few));
//		vector<big> large_input(Kernels::nextPow2(a_lot));
//		for (int i = 0; i < small_input.size(); ++i)
//			small_input[i] = makeBig(i);
//		for (int i = 0; i < large_input.size(); ++i)
//			large_input[i] = makeBig(i);
//		When("we check to see if these numbers are in order in parallel") {
//			cl_int error = 0;
//			cl_int smallResult, largeResult;
//			cl::Buffer b_small_input, b_large_input;
//			error |= CLFW::getBuffer(b_small_input, "small_input", Kernels::nextPow2(a_few) * sizeof(big));
//			error |= CLFW::getBuffer(b_large_input, "large_input", Kernels::nextPow2(a_lot) * sizeof(big));
//			error |= CLFW::Upload<big>(small_input, b_small_input);
//			error |= CLFW::Upload<big>(large_input, b_large_input);
//			error |= CheckBigOrder_p(b_small_input, Kernels::nextPow2(a_few), smallResult);
//			error |= CheckBigOrder_p(b_large_input, Kernels::nextPow2(a_lot), largeResult);
//			Require(error == CL_SUCCESS);
//			Then("we should get back true") {
//				Require(smallResult == 0);
//				Require(largeResult == 0);
//			}
//		}
//		And_when("we modify the list so it isn't in order") {
//			small_input[10] = makeBig(999);
//			large_input[600] = makeBig(13);
//			When("we check to see if these numbers are in order in parallel") {
//				cl_int error = 0;
//				cl_int smallResult, largeResult;
//				cl::Buffer b_small_input, b_large_input;
//				error |= CLFW::getBuffer(b_small_input, "small_input", Kernels::nextPow2(a_few) * sizeof(big));
//				error |= CLFW::getBuffer(b_large_input, "large_input", Kernels::nextPow2(a_lot) * sizeof(big));
//				error |= CLFW::Upload<big>(small_input, b_small_input);
//				error |= CLFW::Upload<big>(large_input, b_large_input);
//				TODO("make checkorder work for non-power of two elements");
//				error |= CheckBigOrder_p(b_small_input, Kernels::nextPow2(a_few), smallResult);
//				error |= CheckBigOrder_p(b_large_input, Kernels::nextPow2(a_lot), largeResult);
//				Require(error == CL_SUCCESS);
//				Then("we should get back the total number of out of order numbers") {
//					Require(smallResult == 1);
//					Require(largeResult == 1);
//				}
//			}
//		}
//	}
//}

/* Predication Kernels */
Scenario("Predicate by bit", "[predication]") {
	Given("N random integers") {
		vector<cl_int>small_input = generateDeterministicRandomIntegers(a_few);
		When("we predicate these integers in series") {
			vector<cl_int>small_output_s(a_few);
			PredicateByBit_s(small_input, 2, 1, small_output_s);
			Then("the predication for each number will be 1 only if the i'th bit matches \"compared\"") {
				int success = true;
				for (int i = 0; i < a_few; ++i)
					success &= ((small_input[i] & (1 << 2)) >> 2 == 1) == small_output_s[i];
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				vector<cl_int>small_output_p(a_few);
				cl::Buffer b_small_input, b_small_output;
				cl_int error = 0;
				error |= CLFW::getBuffer(b_small_input, "small", a_few * sizeof(cl_int));
				error |= CLFW::Upload<cl_int>(small_input, b_small_input);
				error |= PredicateByBit_p(b_small_input, 2, 1, a_few, "a", b_small_output);
				error |= CLFW::Download<cl_int>(b_small_output, a_few, small_output_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i) success &= small_output_p[i] == small_output_s[i];
				Require(success == true);
			}
		}
	}
}
Scenario("Predicate big by bit", "[predication]") {
	Given("N random bigs") {
		vector<big>small_input = generateDeterministicRandomBigs(a_few);
		When("we predicate these integers in series") {
			vector<cl_int>small_output_s(a_few);
			PredicateBUByBit_s(small_input, 2, 1, small_output_s);
			Then("the predication for each number will be 1 only if the i'th bit matches \"compared\"") {
				int success = true;
				for (int i = 0; i < a_few; ++i)
					success &= (getBigBit(&small_input[i], 0, 2) == 1) == small_output_s[i];
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				vector<cl_int>small_output_p(a_few);
				vector<cl_int>large_output_p(a_lot);
				cl::Buffer b_small_input, b_small_output;
				cl_int error = 0;
				error |= CLFW::getBuffer(b_small_input, "small", a_few * sizeof(big));
				error |= CLFW::Upload<big>(small_input, b_small_input);
				error |= PredicateBigByBit_p(b_small_input, 2, 1, a_few, "a", b_small_output);
				error |= CLFW::Download<cl_int>(b_small_output, a_few, small_output_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i) success &= small_output_p[i] == small_output_s[i];
				Require(success == true);
			}
		}
	}
}
Scenario("Predicate Conflict", "[conflict][predication]") {
	Given("N conflicts") {
		vector<Conflict>small_input = generateDeterministicRandomConflicts(a_few);
		When("we predicate these conflicts in series") {
			vector<cl_int>small_output_s(a_few);
			PredicateConflicts_s(small_input, small_output_s);
			Then("the predication for each number will be 1 only if the i'th bit matches \"compared\"") {
				int success = true;
				for (int i = 0; i < a_few; ++i)
					success &= (small_input[i].color == -2) == small_output_s[i];
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				vector<cl_int>small_output_p(a_few);
				cl::Buffer b_small_input, b_small_output;
				cl_int error = 0;
				error |= CLFW::getBuffer(b_small_input, "small", a_few * sizeof(Conflict));
				error |= CLFW::Upload<Conflict>(small_input, b_small_input);
				error |= PredicateConflicts_p(b_small_input, a_few, "a", b_small_output);
				error |= CLFW::Download<cl_int>(b_small_output, a_few, small_output_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i) success &= small_output_p[i] == small_output_s[i];
				Require(success == true);
			}
		}
	}
}

/* Compaction Kernels */
Scenario("Integer Compaction", "[compaction]") {
	Given("N random integers, an arbitrary predication, and an inclusive prefix sum of that predication") {
		vector<cl_int>small_input = generateDeterministicRandomIntegers(a_few);
		vector<cl_int> small_pred(a_few), small_addr(a_few), small_output_s(a_few);
		/* In this example, odd indexes are compacted to the left. */
		for (int i = 0; i < a_few; ++i) { small_pred[i] = i % 2; small_addr[i] = (i + 1) / 2; }
		When("these integers are compacted in series") {
			Compact_s(small_input, small_pred, small_addr, small_output_s);
			Then("elements predicated true are moved to their cooresponding addresses.") {
				int success = true;
				for (int i = 0; i < a_few / 2; ++i) success &= (small_output_s[i] == small_input[(i * 2) + 1]);
				Require(success == true);
			}
			Then("elements predicated false are placed after the last truely predicated element and in their original order.") {
				int success = true;
				for (int i = a_few / 2; i < a_few; ++i) success &= (small_output_s[i] == small_input[(i - (a_few / 2)) * 2]);
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<cl_int> small_output_p(a_few);
				cl::Buffer b_small_input, b_small_pred, b_small_addr, b_small_output;
				error |= CLFW::getBuffer(b_small_input, "b_small_input", a_few * sizeof(cl_int));
				error |= CLFW::getBuffer(b_small_pred, "b_small_pred", a_few * sizeof(cl_int));
				error |= CLFW::getBuffer(b_small_addr, "b_small_addr", a_few * sizeof(cl_int));
				error |= CLFW::getBuffer(b_small_output, "b_small_output", a_few * sizeof(cl_int));
				error |= CLFW::Upload<cl_int>(small_input, b_small_input);
				error |= CLFW::Upload<cl_int>(small_pred, b_small_pred);
				error |= CLFW::Upload<cl_int>(small_addr, b_small_addr);
				error |= Compact_p(b_small_input, b_small_pred, b_small_addr, a_few, b_small_output);
				error |= CLFW::Download<cl_int>(b_small_output, a_few, small_output_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i) success &= (small_output_p[i] == small_output_s[i]);
				Require(success == true);
			}
		}
	}
}
Scenario("Big Unsigned Compaction", "[compaction]") {
	Given("N random bigs, an arbitrary predication, and an inclusive prefix sum of that predication") {
		vector<big>small_input = generateDeterministicRandomBigs(a_few);
		vector<cl_int> small_pred(a_few), small_addr(a_few);
		vector<big> small_output_s(a_few);
		/* In this example, odd indexes are compacted to the left. */
		for (int i = 0; i < a_few; ++i) { small_pred[i] = i % 2; small_addr[i] = (i + 1) / 2; }
		When("these integers are compacted in series") {
			BigCompact_s(small_input, small_pred, small_addr, small_output_s);
			Then("elements predicated true are moved to their cooresponding addresses.") {
				int success = true;
				for (int i = 0; i < a_few / 2; ++i) success &= (compareBig(&small_output_s[i], &small_input[(i * 2) + 1]) == 0);
				Require(success == true);
			}
			Then("elements predicated false are placed after the last truely predicated element and in their original order.") {
				int success = true;
				for (int i = a_few / 2; i < a_few; ++i) success &= (compareBig(&small_output_s[i], &small_input[(i - (a_few / 2)) * 2]) == 0);
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<big> small_output_p(a_few);
				cl::Buffer b_small_input, b_small_pred, b_small_addr, b_small_output;
				error |= CLFW::getBuffer(b_small_input, "b_small_input", a_few * sizeof(big));
				error |= CLFW::getBuffer(b_small_pred, "b_small_pred", a_few * sizeof(cl_int));
				error |= CLFW::getBuffer(b_small_addr, "b_small_addr", a_few * sizeof(cl_int));
				error |= CLFW::getBuffer(b_small_output, "b_small_output", a_few * sizeof(big));
				error |= CLFW::Upload<big>(small_input, b_small_input);
				error |= CLFW::Upload<cl_int>(small_pred, b_small_pred);
				error |= CLFW::Upload<cl_int>(small_addr, b_small_addr);
				error |= BigCompact_p(b_small_input, a_few, b_small_pred, b_small_addr, b_small_output);
				error |= CLFW::Download<big>(b_small_output, a_few, small_output_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i) success &= (compareBig(&small_output_p[i], &small_output_s[i]) == 0);
				Require(success == true);
			}
		}
	}
}
Scenario("Conflict Compaction", "[conflict][compaction]") {
	Given("N random conflicts, an arbitrary predication, and an inclusive prefix sum of that predication") {
		vector<Conflict>small_input = generateDeterministicRandomConflicts(a_few);
		vector<cl_int> small_pred(a_few), small_addr(a_few);
		vector<Conflict>small_output_s(a_few);
		/* In this example, odd indexes are compacted to the left. */
		for (int i = 0; i < a_few; ++i) { small_pred[i] = i % 2; small_addr[i] = (i + 1) / 2; }
		When("these conflicts are compacted in series") {
			CompactConflicts_s(small_input, small_pred, small_addr, small_output_s);
			Then("elements predicated true are moved to their cooresponding addresses.") {
				int success = true;
				for (int i = 0; i < a_few / 2; ++i) success &= (compareConflict(&small_output_s[i], &small_input[(i * 2) + 1]));
				Require(success == true);
			}
			Then("elements predicated false are placed after the last truely predicated element and in their original order.") {
				int success = true;
				for (int i = a_few / 2; i < a_few; ++i) success &= (compareConflict(&small_output_s[i], &small_input[(i - (a_few / 2)) * 2]));
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<Conflict> small_output_p(a_few);
				cl::Buffer b_small_input, b_small_pred, b_small_addr, b_small_output;
				error |= CLFW::getBuffer(b_small_input, "b_small_input", a_few * sizeof(Conflict));
				error |= CLFW::getBuffer(b_small_pred, "b_small_pred", a_few * sizeof(cl_int));
				error |= CLFW::getBuffer(b_small_addr, "b_small_addr", a_few * sizeof(cl_int));
				error |= CLFW::getBuffer(b_small_output, "b_small_output", a_few * sizeof(Conflict));
				error |= CLFW::Upload<Conflict>(small_input, b_small_input);
				error |= CLFW::Upload<cl_int>(small_pred, b_small_pred);
				error |= CLFW::Upload<cl_int>(small_addr, b_small_addr);
				error |= CompactConflicts_p(b_small_input, b_small_pred, b_small_addr, a_few, b_small_output);
				error |= CLFW::Download<Conflict>(b_small_output, a_few, small_output_p);
				Require(error == CL_SUCCESS);
				int success = true;
				for (int i = 0; i < a_few; ++i) success &= (compareConflict(&small_output_p[i], &small_output_s[i]));
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
				cl::Buffer b_small_input, b_large_input, b_small_output, b_large_output;
				cl_int error = 0;
				error |= CLFW::getBuffer(b_small_input, "smallin", a_few * sizeof(cl_int));
				error |= CLFW::getBuffer(b_large_input, "largein", a_lot * sizeof(cl_int));
				error |= CLFW::getBuffer(b_small_output, "smallout", a_few * sizeof(cl_int));
				error |= CLFW::getBuffer(b_large_output, "largeout", a_lot * sizeof(cl_int));
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
Scenario("Four Way Frequency Count (bigs)", "[sort][4way]") {
	Given("a block size and a set of big numbers evenly divisable by that block size ") {
		cl_int blockSize = 4;
		vector<big> in(20);
		in[0].blk[0] = 1; in[1].blk[0] = 2; in[2].blk[0] = 0; in[3].blk[0] = 3;
		in[4].blk[0] = 0; in[5].blk[0] = 1; in[6].blk[0] = 1; in[7].blk[0] = 0;
		in[8].blk[0] = 3; in[9].blk[0] = 3; in[10].blk[0] = 3; in[11].blk[0] = 2;
		in[12].blk[0] = 1; in[13].blk[0] = 2; in[14].blk[0] = 2; in[15].blk[0] = 0;
		in[16].blk[0] = 2; in[17].blk[0] = 0; in[18].blk[0] = 0; in[19].blk[0] = 2;

		writeToFile<big>(in, "TestData//4waysort//original.bin");
		When("we compute the 4 way frequency count in series") {
			vector<big> s_shuffle, result;
			vector<cl_int> localPrefix, s_blockSum, localShuffleAddr;
			BigFourWayPrefixSumWithShuffle_s(in, blockSize, 0, 0, s_shuffle, s_blockSum);
			Then("the results should be valid") {
				vector<big> f_shuffle = readFromFile<big>("TestData//4waysort//shuffle.bin", 20);
				vector<cl_int> f_blockSum = readFromFile<cl_int>("TestData//4waysort//blockSum.bin", 20);
				
				cl_int success = true;
				for (int i = 0; i < 20; ++i) {
					success &= (0 == compareBig(&f_shuffle[i], &s_shuffle[i]));
					success &= (f_blockSum[i] == s_blockSum[i]);
				}

				And_then("the series results should match the parallel results") {
					cl_int error = 0;
					vector<big> p_shuffle;
					vector<cl_int> p_blockSum;
					cl::Buffer b_in, b_blkSum, b_shuffle;
					error |= CLFW::getBuffer(b_in, "in", 20 * sizeof(big));
					error |= CLFW::Upload<big>(in, b_in);
					error |= BigFourWayPrefixSumAndShuffle_p(b_in, 20, blockSize, 0, 0, b_blkSum, b_shuffle);
					error |= CLFW::Download<big>(b_shuffle, 20, p_shuffle);
					error |= CLFW::Download<cl_int>(b_blkSum, 20, p_blockSum);
					Require(error == CL_SUCCESS);

					success = true;
					for (cl_int i = 0; i < 20; ++i) {
						success &= (compareBig(&p_shuffle[i], &s_shuffle[i]) == 0);
						success &= (p_blockSum[i] == s_blockSum[i]);
					}
					Require(success == true);
				}
			}
		}
	}
}
Scenario("Move Four Way Shuffled Elements (bigs)", "[sort][4way]")
{
	Given("the shuffled elements, block sums, and prefix block sums produced by the 4 way frequency count") {
		vector<big> f_shuffle = readFromFile<big>("TestData//4waysort//shuffle.bin", 20);
		vector<cl_int> f_blockSum = readFromFile<cl_int>("TestData//4waysort//blockSum.bin", 20);
		vector<cl_int> f_prefixBlockSum = readFromFile<cl_int>("TestData//4waysort//prefixBlockSum.bin", 20);
		When("we use the block sum and prefix block sum to move these shuffled elements in series") {
			vector<big> s_result;
			MoveBigElements_s(f_shuffle, f_blockSum, f_prefixBlockSum, 4, 0, 0, s_result);
			Then("the results should be valid") {
				vector<big> f_result = readFromFile<big>("TestData//4waysort//result.bin", 20);
				cl_int success = true;
				for (cl_int i = 0; i < 20; ++i) {
					success &= (compareBig(&s_result[i], &f_result[i]));
				}
				And_then("the series results should match the parallel results") {
					cl_int error = 0;
					vector<big> p_result;
					vector<cl_int> p_blockSum;
					cl::Buffer b_shuffle, b_blkSum, b_prefixBlkSum, b_result;
					error |= CLFW::getBuffer(b_shuffle, "shuffle", 20 * sizeof(big));
					error |= CLFW::getBuffer(b_blkSum, "blkSum", 20 * sizeof(cl_int));
					error |= CLFW::getBuffer(b_prefixBlkSum, "prefixBlkSum", 20 * sizeof(cl_int));
					error |= CLFW::Upload<big>(f_shuffle, b_shuffle);
					error |= CLFW::Upload<cl_int>(f_blockSum, b_blkSum);
					error |= CLFW::Upload<cl_int>(f_prefixBlockSum, b_prefixBlkSum);
					error |= MoveBigElements_p(b_shuffle, 20, b_blkSum, b_prefixBlkSum, 4, 0, 0, b_result);
					error |= CLFW::Download<big>(b_result, 20, p_result);
					Require(error == CL_SUCCESS);

					success = true;
					for (cl_int i = 0; i < 20; ++i) {
						success &= (compareBig(&p_result[i], &s_result[i]) == 0);
					}
					Require(success == true);
				}
			}
		}
	}
}
Scenario("Four Way Radix Sort (bigs)", "[sort][4way]") {
	Given("an unsorted set of big") {
		cl_int numElements = a_lot;
		vector<big> input(numElements);
		for (int i = 0; i < numElements; ++i) {
			input[i] = { (cl_ulong)(numElements - i), 0};
		}
		When("we sort that data using the parallel 4 way radix sorter") {
			vector<big> result;
			vector<big> result2;
			cl_int error = 0;
			cl::Buffer b_input, b_other;
			error |= CLFW::getBuffer(b_input, "input", numElements * sizeof(big));
			error |= CLFW::Upload<big>(input, b_input);
			error |= RadixSortBig_p(b_input, numElements, 48, "");
			error |= CLFW::Download<big>(b_input, numElements, result);

			Require(error == CL_SUCCESS);

			Then("the results should be valid") {
				cl_int success = true;
				for (int i = 0; i < a_lot; ++i) {
					big temp = {i + 1, 0};
					success &= (compareBig(&result[i], &temp) == 0);
				}
				Require(success == true);
			}
		}
	}
}
Scenario("Four Way Radix Sort (<big, cl_int> by Key)", "[sort][4way]") {
	Given("An arbitrary set of unsigned key and integer value pairs") {
		vector<big> keys(a_lot);
		vector<cl_int> values(a_lot);
		for (int i = 0; i < a_lot; ++i) {
			keys[i] = makeBig(a_lot - i);
			values[i] = a_lot - i;
		}

		When("these pairs are sorted by key in parallel") {
			cl_int error = 0;
			cl::Buffer b_keys, b_values;
			error |= CLFW::getBuffer(b_keys, "b_keys", a_lot * sizeof(big));
			error |= CLFW::getBuffer(b_values, "b_values", a_lot * sizeof(cl_int));
			error |= CLFW::Upload<big>(keys, b_keys);
			error |= CLFW::Upload<cl_int>(values, b_values);
			Require(error == CL_SUCCESS);
			error |= RadixSortBigToInt_p(b_keys, b_values, a_lot, 20, "");
			Require(error == CL_SUCCESS);
			Then("The key value pairs are ordered by keys assending") {
				vector<big> keys_out_p(a_lot);
				vector<cl_int> values_out_p(a_lot);
				error |= CLFW::Download<big>(b_keys, a_lot, keys_out_p);
				error |= CLFW::Download<cl_int>(b_values, a_lot, values_out_p);
				int success = true;
				for (int i = 0; i < a_lot; ++i) {
					success &= (values_out_p[i] && values_out_p[i] == i + 1);
					big temp = makeBig(i + 1);
					success &= (compareBig(&keys_out_p[i], &temp) == 0);
				}
				Require(success == true);
			}
		}
	}
}
Scenario("Four Way Radix Sort (<cl_int, cl_int> by Key)", "[sort][4way]") {
	Given("An arbitrary set of unsigned key and integer value pairs") {
		vector<cl_int> keys(a_lot);
		vector<cl_int> values(a_lot);
		for (int i = 0; i < a_lot; ++i) {
			keys[i] = a_lot - i;
			values[i] = a_lot - i;
		}

		When("these pairs are sorted by key in parallel") {
			cl_int error = 0;
			cl::Buffer b_keys, b_values;
			error |= CLFW::getBuffer(b_keys, "b_keys", a_lot * sizeof(cl_int));
			error |= CLFW::getBuffer(b_values, "b_values", a_lot * sizeof(cl_int));
			error |= CLFW::Upload<cl_int>(keys, b_keys);
			error |= CLFW::Upload<cl_int>(values, b_values);
			Require(error == CL_SUCCESS);
			error |= RadixSortIntToInt_p(b_keys, b_values, a_lot, 20, "");
			Require(error == CL_SUCCESS);
			Then("The key value pairs are ordered by keys assending") {
				vector<cl_int> keys_out_p(a_lot);
				vector<cl_int> values_out_p(a_lot);
				error |= CLFW::Download<cl_int>(b_keys, a_lot, keys_out_p);
				error |= CLFW::Download<cl_int>(b_values, a_lot, values_out_p);
				int success = true;
				for (int i = 0; i < a_lot; ++i) {
					success &= (values_out_p[i] && values_out_p[i] == i + 1);
					success &= (keys_out_p[i] && keys_out_p[i] == i + 1);
				}
				Require(success == true);
			}
		}
	}
}
//Scenario("Parallel Radix Sort", "[1][sort][integration][failing][disabled]") {
//	Given("An arbitrary set of numbers") {
//		vector<cl_ulong> small_input(a_few);
//		vector<cl_ulong> large_input(a_lot);
//
//		for (cl_ulong i = 0; i < a_few; ++i) small_input[i] = a_few - i;
//		for (cl_ulong i = 0; i < a_lot; ++i) large_input[i] = a_lot - i;
//
//		When("these numbers are sorted in parallel") {
//			cl_int error = 0;
//			cl::Buffer b_small_input, b_large_input;
//			error |= CLFW::getBuffer(b_small_input, "b_small_input", a_few * sizeof(cl_ulong));
//			error |= CLFW::getBuffer(b_large_input, "b_large_input", a_lot * sizeof(cl_ulong));
//			error |= CLFW::Upload<cl_ulong>(small_input, b_small_input);
//			error |= CLFW::Upload<cl_ulong>(large_input, b_large_input);
//			error |= OldRadixSort_p(b_small_input, a_few, 20);
//			error |= OldRadixSort_p(b_large_input, a_lot, 20);
//			Require(error == CL_SUCCESS);
//			Then("the numbers are ordered assending") {
//				vector<cl_ulong> small_output_p(a_few), large_output_p(a_lot);
//				error |= CLFW::Download<cl_ulong>(b_small_input, a_few, small_output_p);
//				error |= CLFW::Download<cl_ulong>(b_large_input, a_lot, large_output_p);
//				int success = true;
//				for (cl_ulong i = 0; i < a_few; ++i) {
//					big temp = makeBig(i + 1);
//					success &= (small_output_p[i] == i + 1);
//				}
//				Require(success == true);
//				for (cl_ulong i = 0; i < a_lot; ++i) {
//					success &= (large_output_p[i] == i + 1);
//				}
//				Require(success == true);
//			}
//		}
//	}
//}
Scenario("Parallel Radix Sort (Pairs by Key)", "[2][sort][integration]") {
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
			error |= CLFW::getBuffer(b_small_keys, "b_small_keys", a_few * sizeof(cl_int));
			error |= CLFW::getBuffer(b_small_values, "b_small_values", a_few * sizeof(cl_int));
			error |= CLFW::getBuffer(b_large_keys, "b_large_keys", a_lot * sizeof(cl_int));
			error |= CLFW::getBuffer(b_large_values, "b_large_values", a_lot * sizeof(cl_int));
			error |= CLFW::Upload<cl_int>(small_keys_in, b_small_keys);
			error |= CLFW::Upload<cl_int>(small_values_in, b_small_values);
			error |= CLFW::Upload<cl_int>(large_keys_in, b_large_keys);
			error |= CLFW::Upload<cl_int>(large_values_in, b_large_values);
			Require(error == CL_SUCCESS);
			error |= OldRadixSortPairsByKey(b_small_keys, b_small_values, a_few);
			Require(error == CL_SUCCESS);
			error |= OldRadixSortPairsByKey(b_large_keys, b_large_values, a_lot);
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
			When("these points are quantized in series") {
				vector<int2> small_output_s(a_few);
				QuantizePoints_s(small_input, bb, resolution_width, small_output_s);
				Then("the series results match the parallel results") {
					cl_int error = 0;
					cl::Buffer b_small_input, b_small_output;
					vector<intn> small_output_p(a_few);
					error |= CLFW::getBuffer(b_small_input, "b_small_input", a_few * sizeof(int2));
					error |= CLFW::Upload<floatn>(small_input, b_small_input);
					error |= QuantizePoints_p(b_small_input, a_few, bb, resolution_width, "a", b_small_output);
					error |= CLFW::Download<intn>(b_small_output, a_few, small_output_p);
					Require(error == CL_SUCCESS);
					int success = true;
					for (int i = 0; i < a_few; ++i) {
						success &= (small_output_s[i] == small_output_p[i]);
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
			vector<big> small_output_s(a_few);
			When("these points are placed on a Z-Order curve in series") {
				QPointsToZPoints_s(small_input, 30, small_output_s);
				Then("the series results match the parallel results") {
					cl_int error = 0;
					vector<big> small_output_p(a_few), large_output_p(a_lot);
					cl::Buffer b_small_input, b_small_output;
					error |= CLFW::getBuffer(b_small_input, "b_small_input", a_few * sizeof(big));
					error |= CLFW::Upload<intn>(small_input, b_small_input);
					error |= QPointsToZPoints_p(b_small_input, a_few, 30, "a", b_small_output);
					error |= CLFW::Download<big>(b_small_output, a_few, small_output_p);
					Require(error == CL_SUCCESS);
					int success = true;
					for (int i = 0; i < a_few; ++i) success &= (compareBig(&small_output_s[i], &small_output_p[i]) == 0);
					Require(success == true);
				}
			}
		}
	}
}

/* Unique Kernels */
Scenario("Unique Sorted big", "[sort][unique]") {
	Given("An ascending sorted set of bigs") {
		vector<big> small_zpoints = readFromFile<big>("TestData//few_non-unique_s_zpoints.bin", a_few);
		When("those bigs are uniqued in parallel") {
			cl_int error = 0, newSmallSize, newLargeSize;
			cl::Buffer b_small_zpoints, b_unique_small_zpoints;
			error |= CLFW::getBuffer(b_small_zpoints, "b_small_zpoints", a_few * sizeof(big));
			error |= CLFW::Upload<big>(small_zpoints, b_small_zpoints);
			error |= UniqueSorted(b_small_zpoints, a_few, "a", newSmallSize);
			vector<big> p_small_zpoints(newSmallSize);
			error |= CLFW::Download<big>(b_small_zpoints, newSmallSize, p_small_zpoints);
			Then("the resulting set should match the uniqued series set.") {
				auto sm_last = unique(small_zpoints.begin(), small_zpoints.end(), weakEqualsBig);
				small_zpoints.erase(sm_last, small_zpoints.end());
				Require(small_zpoints.size() == newSmallSize);
				cl_int success = 1;
				for (int i = 0; i < small_zpoints.size(); ++i)
					success &= (compareBig(&small_zpoints[i], &p_small_zpoints[i]) == 0);
				Require(success == true);
			}
		}
	}
}
Scenario("Unique Sorted big color pairs", "[sort][unique]") {
	Given("An ascending sorted set of bigs") {
		vector<big> small_keys(a_few);
		vector<cl_int> small_values(a_few);

		for (int i = 0; i < a_few; ++i) {
			small_values[i] = i / 2;
			small_keys[i] = makeBig(i / 2);
		}

		When("those bigs are uniqued in parallel") {
			cl_int error = 0, newSmallSize, newLargeSize;
			cl::Buffer b_small_keys, b_unique_small_keys;
			cl::Buffer b_small_values, b_unique_small_values;
			error |= CLFW::getBuffer(b_small_keys, "b_small_zpoints", a_few * sizeof(big));
			error |= CLFW::getBuffer(b_small_values, "b_small_values", a_few * sizeof(cl_int));
			error |= CLFW::Upload<big>(small_keys, b_small_keys);
			error |= CLFW::Upload<cl_int>(small_values, b_small_values);
			error |= UniqueSortedBUIntPair(b_small_keys, b_small_values, a_few, "a", newSmallSize);
			vector<big> p_small_zpoints(newSmallSize);
			error |= CLFW::Download<big>(b_small_keys, newSmallSize, p_small_zpoints);
			Then("the resulting set should match the uniqued series set.") {
				auto sm_last = unique(small_keys.begin(), small_keys.end(), weakEqualsBig);
				small_keys.erase(sm_last, small_keys.end());
				Require(small_keys.size() == newSmallSize);
				cl_int success = 1;
				for (int i = 0; i < small_keys.size(); ++i)
					success &= (compareBig(&small_keys[i], &p_small_zpoints[i]) == 0);
				Require(success == true);
			}
		}
	}
}

/* Tree Building Kernels */
Scenario("Build Binary Radix Tree", "[tree]") {
	Given("A set of unique ordered zpoints") {
		cl_int numPts = readFromFile<cl_int>("TestData//simple//uNumPts.bin");
		vector<big> zpnts = readFromFile<big>("TestData//simple//usZPts.bin", numPts);
		Resln resln = readFromFile<Resln>("TestData//simple//resln.bin");

		When("we build a binary radix tree using these points in series") {
			vector<BrtNode> s_brt;
			BuildBinaryRadixTree_s(zpnts, resln.mbits, s_brt);

			Then("the results should be valid") {
				vector<BrtNode> f_brt = readFromFile<BrtNode>("TestData//simple//brt.bin", numPts - 1);
				cl_int success = 1;
				for (int i = 0; i < numPts - 1; ++i) {
					success &= (true == compareBrtNode(&s_brt[i], &f_brt[i]));
				}
				Require(success == true);
			}

			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<BrtNode> p_brt;
				cl::Buffer b_zpoints, b_brt;
				error |= CLFW::getBuffer(b_zpoints, "b_zpoints", numPts * sizeof(big));
				error |= CLFW::Upload<big>(zpnts, b_zpoints);
				error |= BuildBinaryRadixTree_p(b_zpoints, numPts, resln.mbits, "a", b_brt);
				error |= CLFW::Download<BrtNode>(b_brt, numPts - 1, p_brt);
				Require(error == CL_SUCCESS);
				Then("the resulting binary radix tree should be valid") {
					/* Precomputed */
					int success = true;
					for (int i = 0; i < numPts - 1; ++i) {
						success &= (true == compareBrtNode(&s_brt[i], &p_brt[i]));
					}
					Require(success == true);
				}
			}
		}
	}
}
Scenario("Build Colored Binary Radix Tree", "[tree]") {
	Given("A set of colored unique ordered zpoints") {
		cl_int numPts = readFromFile<cl_int>("TestData//simple//uNumPts.bin");
		vector<big> zpnts = readFromFile<big>("TestData//simple//usZPts.bin", numPts);
		Resln resln = readFromFile<Resln>("TestData//simple//resln.bin");
		vector<cl_int> leafColors = readFromFile<cl_int>("TestData//simple//uniqueColors.bin", numPts);
		
		When("we build a colored binary radix tree using these points in series") {
			vector<BrtNode> brt_s;
			vector<cl_int> brtColors_s;
			BuildColoredBinaryRadixTree_s(zpnts, leafColors, resln.mbits, brt_s, brtColors_s);
			Then("the resulting binary radix tree and cooresponding colors should be valid") {
				vector<BrtNode> brt_f = readFromFile<BrtNode>("TestData//simple//brt.bin", numPts - 1);
				vector<cl_int> brtColors_f = readFromFile<cl_int>("TestData//simple//unpropagatedBrtColors.bin", numPts - 1);
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
				error |= CLFW::getBuffer(b_zpoints, "b_zpoints", numPts * sizeof(big));
				error |= CLFW::getBuffer(b_leafColors, "b_leafColors", numPts * sizeof(cl_int));
				error |= CLFW::Upload<big>(zpnts, b_zpoints);
				error |= CLFW::Upload<cl_int>(leafColors, b_leafColors);
				error |= BuildColoredBinaryRadixTree_p(b_zpoints, b_leafColors, numPts, resln.mbits, "", b_brt, b_brtColors);
				error |= CLFW::Download<BrtNode>(b_brt, numPts - 1, brt_p);
				error |= CLFW::Download<cl_int>(b_brtColors, numPts - 1, brtColors_p);
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
Scenario("Propagate Brt Colors", "[tree]") {
	Given("a colored binary radix tree") {
		cl_int totalPoints = readFromFile<cl_int>("TestData//simple//numPoints.bin");
		vector<BrtNode> brt = readFromFile<BrtNode>("TestData//simple//brt.bin", totalPoints - 1);
		vector<cl_int> brtColors_s = readFromFile<cl_int>("TestData//simple//unpropagatedBrtColors.bin", totalPoints - 1);
		vector<cl_int> brtColors_f = readFromFile<cl_int>("TestData//simple//unpropagatedBrtColors.bin", totalPoints - 1);

		When("we propagate the BRT colors up the tree in series") {
			PropagateBRTColors_s(brt, brtColors_s);
			Then("the results should be valid") {
				vector<cl_int> brtColors_f = readFromFile<cl_int>("TestData//simple//brtColors.bin", totalPoints - 1);

				cl_int success = true;
				for (cl_int i = 0; i < totalPoints - 1; i++) {
					success &= brtColors_s[i] == brtColors_f[i];
				}
				Require(success == true);
			}
			Then("the series results should match the parallel results") {
				cl_int error = 0;
				cl::Buffer b_brt, b_brtColors;
				vector<cl_int> brtColors_p;
				error |= CLFW::getBuffer(b_brt, "brt", (totalPoints - 1) * sizeof(BrtNode));
				error |= CLFW::getBuffer(b_brtColors, "b_brtColors", (totalPoints - 1) * sizeof(cl_int));
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
		cl_int numPts = readFromFile<cl_int>("TestData//simple//uNumPts.bin");
		auto brt = readFromFile<BrtNode>("TestData//simple//brt.bin", numPts - 1);
		When("we use that binary radix tree to build an octree in parallel") {
			cl_int error = 0, octree_size;
			cl::Buffer b_brt, b_octree, nullBuffer;
			error |= CLFW::getBuffer(b_brt, "brt", (numPts - 1) * sizeof(BrtNode));
			error |= CLFW::Upload<BrtNode>(brt, b_brt);
			error |= BinaryRadixToOctree_p(b_brt, false, nullBuffer, numPts, "", b_octree, octree_size);
			vector<OctNode> p_octree(octree_size);
			error |= CLFW::Download<OctNode>(b_octree, octree_size, p_octree);
			
			Then("our results should be valid") {
				/*Require(small_octree_size == 11);
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
				Require(success == true);*/
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
		cl_int numOctNodes = readFromFile<cl_int>("TestData//simple//numOctNodes.bin");
		auto octree = readFromFile<OctNode>("TestData//simple//octree.bin", numOctNodes);

		When("we generate the leaves of this octree in series") {
			vector<cl_int> pred_s(4 * numOctNodes);
			vector<Leaf> leaves_s(4 * numOctNodes);
			GenerateLeaves_s(octree, numOctNodes, leaves_s, pred_s);
			Then("the parallel results match the serial ones") {
				cl_int error = 0;
				vector<cl_int> pred_p(4* numOctNodes);
				vector<Leaf> leaves_p(4* numOctNodes);
				cl::Buffer b_small_octree, b_small_pred, b_small_leaves;
				error |= CLFW::getBuffer(b_small_octree, "b_small_octree", 4 * numOctNodes * sizeof(OctNode));
				error |= CLFW::Upload<OctNode>(octree, b_small_octree);
				error |= GenerateLeaves_p(b_small_octree, numOctNodes, b_small_leaves, b_small_pred);
				error |= CLFW::Download(b_small_pred, 4 * numOctNodes, pred_p);
				error |= CLFW::Download(b_small_leaves, 4 * numOctNodes, leaves_p);
				Require(error == CL_SUCCESS);
				cl_int success = true;
				for (int i = 0; i < 4 * 11; ++i) {
					success &= (compareLeaf(&leaves_s[i], &leaves_p[i]));
					success &= (pred_s[i] == pred_p[i]);
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
		vector<big> p(2);
		p[0] = makeBig(240); //11110000
		p[1] = makeBig(243); //11110011
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
		vector<big> small_zpoints = readFromFile<big>("TestData//few_u_s_zpoints.bin", a_few);
		vector<big> large_zpoints = readFromFile<big>("TestData//lot_u_s_zpoints.bin", a_lot);
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
				error |= CLFW::getBuffer(b_small_lines, "b_small_lines", a_few * sizeof(Line));
				error |= CLFW::getBuffer(b_large_lines, "b_large_lines", a_lot * sizeof(Line));
				error |= CLFW::getBuffer(b_small_zpoints, "b_small_zpoints", a_few * sizeof(big));
				error |= CLFW::getBuffer(b_large_zpoints, "b_large_zpoints", a_lot * sizeof(big));
				error |= CLFW::Upload<Line>(small_lines, b_small_lines);
				error |= CLFW::Upload<Line>(large_lines, b_large_lines);
				error |= CLFW::Upload<big>(small_zpoints, b_small_zpoints);
				error |= CLFW::Upload<big>(large_zpoints, b_large_zpoints);
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
			vector<cl_int> f_LCPToOctnode = readFromFile<cl_int>("TestData//simple//LCPToOctNode.bin", numLines);
			vector<cl_int> s_LCPToOctnode;
			LookUpOctnodeFromLCP_s(lineLCPs, octree, s_LCPToOctnode);
			Then("the series results should be valid") {
				cl_int success = 1;
				for (int i = 0; i < numLines; ++i)
					success &= (f_LCPToOctnode[i] == s_LCPToOctnode[i]);
				Require(success == true);
			}
			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<cl_int> p_LCPToOctnode(numLines);
				cl::Buffer b_octree, b_lineLCPs, b_LCPToOctnode;
				error |= CLFW::getBuffer(b_octree, "b_octree", numOctNodes * sizeof(OctNode));
				error |= CLFW::getBuffer(b_lineLCPs, "b_lineLCPs", nextPow2(numOctNodes * sizeof(LCP)));
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
					error |= CLFW::getBuffer(b_LCPToOctNode, "b_LCPToOctNode", sizeof(cl_int) * numLines);
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
		Resln f_resln											=		readFromFile<Resln>("TestData//simple//resln.bin");
		vector<OctNode> f_octree					=		readFromFile<OctNode>("TestData//simple//octree.bin",						f_numOctnodes);
		vector<Leaf> f_leaves							=		readFromFile<Leaf>("TestData//simple//leaves.bin",							f_numLeaves);
		vector<cl_int> f_LCPToLine				=		readFromFile<cl_int>("TestData//simple//LCPToLine.bin",					f_numLines);
		vector<Pair> f_LCPBounds					=		readFromFile<Pair>("TestData//simple//LCPBounds.bin",						f_numOctnodes);
		vector<Line> f_lines							=		readFromFile<Line>("TestData//simple//lines.bin",								f_numLines);
		vector<intn> f_qpoints						=		readFromFile<intn>("TestData//simple//qpoints.bin",							f_numPoints);
		vector<Conflict> f_conflicts			=		readFromFile<Conflict>("TestData//simple//sparseConflicts.bin",	f_numLeaves);

		When("we use this data to find conflict cells in series") {
			vector<Conflict> s_conflicts;
			FindConflictCells_s(f_octree, f_leaves, f_LCPToLine, f_LCPBounds, f_lines, f_qpoints, f_resln.width, false, s_conflicts);
			Then("the series results match the parallel results") {
					cl_int error = 0;
					vector<Conflict> p_conflicts(f_numLeaves);
					cl::Buffer b_octree, b_leaves, b_LCPToLine, b_LCPBounds, b_lines, b_qpoints, b_conflicts;
					error |= CLFW::getBuffer(b_octree, "b_octree", sizeof(OctNode) * f_numOctnodes);
					error |= CLFW::getBuffer(b_leaves, "b_leaves", sizeof(Leaf) * f_numLeaves);
					error |= CLFW::getBuffer(b_LCPToLine, "b_LCPToLine", sizeof(cl_int) * f_numLines);
					error |= CLFW::getBuffer(b_LCPBounds, "b_LCPBounds", sizeof(Pair) * f_numOctnodes);
					error |= CLFW::getBuffer(b_lines, "b_lines", sizeof(Line) * f_numLines);
					error |= CLFW::getBuffer(b_qpoints, "b_qpoints", sizeof(intn) * f_numPoints);
					error |= CLFW::Upload(f_octree, b_octree);
					error |= CLFW::Upload(f_leaves, b_leaves);
					error |= CLFW::Upload(f_LCPToLine, b_LCPToLine);
					error |= CLFW::Upload(f_LCPBounds, b_LCPBounds);
					error |= CLFW::Upload(f_lines, b_lines);
					error |= CLFW::Upload(f_qpoints, b_qpoints);
					error |= FindConflictCells_p(b_octree, b_leaves, f_numLeaves, b_LCPToLine, b_LCPBounds, 
						b_lines, f_numLines, b_qpoints, f_resln.width, false, b_conflicts);
					error |= CLFW::Download(b_conflicts, f_numLeaves, p_conflicts);
					Require(error == 0);
					cl_int success = true;
					for (int i = 0; i < f_numLeaves; i++) {
						success &= compareConflict(&p_conflicts[i], &s_conflicts[i]);
					}
					Require(success == true);
				}
			Then("the results are valid") {
				cl_int success = true;
				for (int i = 0; i < f_numLeaves; ++i) {
					success &= compareConflict(&s_conflicts[i], &f_conflicts[i]);
					if (!success) {
						success &= compareConflict(&s_conflicts[i], &f_conflicts[i]);
						if (!success)
							success &= compareConflict(&s_conflicts[i], &f_conflicts[i]);
					}
				}
				Require(success == true);
			}
		}
	}
}

/* Ambiguous cell resolution kernels */
Scenario("Sample required resolution points", "[selected][resolution]") {
	Given("a set of conflicts and the quantized points used to build the original octree") {
		cl_int numConflicts = readFromFile<cl_int>("TestData//simple//numConflicts.bin");
		cl_int numPoints = readFromFile<cl_int>("TestData//simple//numPoints.bin");
		vector<Conflict> conflicts = readFromFile<Conflict>("TestData//simple//conflicts.bin", numConflicts);
		vector<intn> qpoints = readFromFile<intn>("TestData//simple//qpoints.bin",  numPoints);
		
		When("we sample the information required to resolve these conflicts in series") {
			vector<ConflictInfo> conflictInfo_s;
			vector<cl_int> numPtsPerConflict_s;
			GetResolutionPointsInfo_s(conflicts, qpoints, conflictInfo_s, numPtsPerConflict_s);
			Then("the series results match the parallel results") {
				cl_int error = 0;
				vector<ConflictInfo> conflictInfo_p(numConflicts);
				vector<cl_int> numPtsPerConflict_p(numConflicts);
				cl::Buffer b_conflicts, b_qpoints, b_conflictInfo, b_numPtsPerConflict;
				error |= CLFW::getBuffer(b_conflicts, "b_conflicts", numConflicts * sizeof(Conflict));
				error |= CLFW::getBuffer(b_qpoints, "b_qpoints", numPoints * sizeof(intn));
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
					}
				}
				Require(success == true);
			}
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
				error |= CLFW::getBuffer(b_scannedNumPtsPerConflict, "b_scannedNumPtsPerConflict", numConflicts * sizeof(cl_int));
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
				error |= CLFW::getBuffer(b_conflicts, "b_conflicts", numConflicts * sizeof(Conflict));
				error |= CLFW::getBuffer(b_conflictInfo, "b_conflictInfo", numConflicts * sizeof(ConflictInfo));
				error |= CLFW::getBuffer(b_scannedNumPtsPerConflict, "b_scannedNumPtsPerConflict", numConflicts * sizeof(cl_int));
				error |= CLFW::getBuffer(b_pntToConflict, "b_pntToConflict", numResPts * sizeof(cl_int));
				error |= CLFW::getBuffer(b_qpoints, "b_qpoints", numPts * sizeof(intn));
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

/* Recursive kernel test */
Scenario("Recursive Dynamic Parallelizm", "[test]") {
	Kernels::DynamicParallelsim();
}