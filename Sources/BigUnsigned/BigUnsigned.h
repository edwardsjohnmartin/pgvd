#pragma once

#ifndef OpenCL
#include <stdbool.h>
#include <stdio.h>
#endif

// Make sure we have NULL.
#ifndef NULL
#define NULL 0
#endif

#define BIG_INTEGER_SIZE 11 //6 results in 32 byte BUs.
typedef int Index; // Type for the index of a block in the array
typedef unsigned char Blk;  // Type for the blocks

// Number of bits in a block.
#if defined(OpenCL)
#define numBUBits (8 * sizeof(Blk))
#else
#define numBUBits (8 * sizeof(Blk))
#endif

// BigUnsigned allows storing integers larger than a long using an array of blk.
typedef struct {
  Index len;                                      // Actual length of the value stored (in blocks)
  Blk blk[BIG_INTEGER_SIZE];
} BigUnsigned;

//~~HELPER FUNCTIONS~~//
// Decreases len to eliminate any leading zero blocks.
inline void zapLeadingZeros(BigUnsigned * bu) {
  while (bu->len > 0 && bu->blk[bu->len - 1] == 0) {
    bu->len--;
  }
}
inline int isBUZero(BigUnsigned *bu) {
  return bu->len == 0;
}
inline void printBUSize() {
  printf("%d\n", sizeof(BigUnsigned));
}

//~~INITIALIZERS~~//
inline int initBUBU(BigUnsigned *result, BigUnsigned *x) {
  if (!result || !x)
    return -1;
  //result->isNULL = x->isNULL;
  result->len = x->len;
  //result->cap = x->cap;
  for (Index i = 0; i < x->len; i++)
    result->blk[i] = x->blk[i];
  return 0;
}
inline int initBU(BigUnsigned *result) {
  if (!result) {
    return -1;
  }
  //result->cap = 0;
  result->len = 0;
  return 0;
}
inline int initBlkBU(BigUnsigned *result, Blk x) {
  if (x < 0) {
    return -1;
  }
  else {
    if (x == 0)
      return initBU(result);
    else {
      //result->cap = 1;
      result->len = 1;
      result->blk[0] = x;
      return 0;
    }
  }
}
//ASSUMES sizeof Blk = sizeof unsigned char!!!
inline int initLongLongBU(BigUnsigned *result, long long x) {
  if (x < 0) {
    return -1;
  }
  else {
    if (x == 0)
      return initBU(result);
    else {
      //result->cap = 8;
      result->len = 8;
      for (int i = 0; i < 8; ++i) {
        result->blk[i] = x;
        x = x >> 8;
      }
      zapLeadingZeros(result);
      return 0;
    }
  }
}
inline int initMorton(BigUnsigned *result, Blk x) {
  return initBlkBU(result, x);
}

//~~BIT/BLOCK ACCESSORS~~//
inline Blk getBUBlock(BigUnsigned *bu, Index i) {
  return i >= bu->len ? 0 : bu->blk[i];
}
inline void setBUBlock(BigUnsigned *bu, Index i, Blk newBlock) {
  if (newBlock == 0) {
    if (i < bu->len) {
      bu->blk[i] = 0;
      zapLeadingZeros(bu);
    }                       // If i >= len, no effect.
  }
  else {
    if (i >= bu->len) {      // The nonzero block extends the number.
      for (Index j = bu->len; j < i; j++)
        bu->blk[j] = 0;
      bu->len = i + 1;
    }
    bu->blk[i] = newBlock;
  }
}
inline Blk getShiftedBUBlock(BigUnsigned *num, Index x, int y) {
  Blk part1 = (x == 0 || y == 0) ? 0 : (num->blk[x - 1] >> (numBUBits - y));
  Blk part2 = (x == num->len) ? 0 : (num->blk[x] << y);
  return part1 | part2;
}
inline bool getBUBit(BigUnsigned *bu, Index bi) {
  Blk blockIndex = bi / numBUBits;
  Blk bitIndex = bi % numBUBits;
  return (getBUBlock(bu, blockIndex) & (1 << bitIndex)) != 0;
}
inline void setBUBit(BigUnsigned *bu, Index bi, bool newBit) {
  Index blockI = bi / numBUBits;
  Blk b = 1;
  Blk block = getBUBlock(bu, blockI), mask = b << (bi % numBUBits);
  block = newBit ? (block | mask) : (block & ~mask);
  setBUBlock(bu, blockI, block);
}

//~~COMPARISON~~//
inline int compareBU(BigUnsigned *x, BigUnsigned *y) {
  // A bigger length implies a bigger number.
  if (x->len < y->len)
    return -1;
  /*CmpRes x = less;*/
  else if (x->len > y->len)
    return 1;
  else {
    // Compare blocks one by one from left to right.
    Index i = x->len;
    while (i > 0) {
      i--;
      if (x->blk[i] == y->blk[i])
        continue;
      else if (x->blk[i] > y->blk[i])
        return 1;
      else
        return -1;
    }
    // If no blocks differed, the numbers are equal.
    return 0;
  }
}
inline int weakCompareBU(BigUnsigned x, BigUnsigned y) {
  // A bigger length implies a bigger number.
  if (x.len < y.len)
    return 0;
  /*CmpRes x = less;*/
  else if (x.len > y.len)
    return 1;
  else {
    // Compare blocks one by one from left to right.
    Index i = x.len;
    while (i > 0) {
      i--;
      if (x.blk[i] == y.blk[i])
        continue;
      else if (x.blk[i] > y.blk[i])
        return 1;
      else
        return 0;
    }
    // If no blocks differed, the numbers are equal.
    return 0;
  }
}
inline bool weakEqualsBU(BigUnsigned x, BigUnsigned y) {
  // A bigger length implies a bigger number.
  if (x.len < y.len)
    return 0;
  /*CmpRes x = less;*/
  else if (x.len > y.len)
    return 0;
  else {
    // Compare blocks one by one from left to right.
    Index i = x.len;
    while (i > 0) {
      i--;
      if (x.blk[i] == y.blk[i])
        continue;
      else if (x.blk[i] > y.blk[i])
        return 0;
      else
        return 0;
    }
    // If no blocks differed, the numbers are equal.
    return 1;
  }
}

//~~ARITHMATIC OPERATIONS~~//
inline int addBU(BigUnsigned *result, BigUnsigned *a, BigUnsigned *b) {
  BigUnsigned copy;
  if (a == result) {
    initBUBU(&copy, a);
    a = &copy;
  }
  if (b == result) {
    initBUBU(&copy, b);
    b = &copy;
  }
  if (a->len == 0) {
    return initBUBU(result, b); //Copy B, return that.
  }
  else if (b->len == 0) {
    return initBUBU(result, a); //Copy A, return that.
  }

  // Some variables...
  // Carries in and out of an addition stage
  bool carryIn, carryOut;
  Blk temp;
  Index i;
  // a2 points to the longer input, b2 points to the shorter
  const BigUnsigned *a2, *b2;
  if (a->len >= b->len) {
    a2 = a;
    b2 = b;
  }
  else {
    a2 = b;
    b2 = a;
  }
  // Set prelimiary length and make room in this BigUnsigned
  result->len = a2->len + 1;
  // For each block index that is present in both inputs...
  for (i = 0, carryIn = false; i < b2->len; i++) {
    // Add input blocks
    temp = a2->blk[i] + b2->blk[i];
    // If a rollover occurred, the result is less than either input.
    // This test is used many times in the BigUnsigned code.
    carryOut = (temp < a2->blk[i]);
    // If a carry was input, handle it
    if (carryIn) {
      temp++;
      carryOut |= (temp == 0);
    }
    result->blk[i] = temp; // Save the addition result
    carryIn = carryOut; // Pass the carry along
  }
  // If there is a carry left over, increase blocks until
  // one does not roll over.
  for (; i < a2->len && carryIn; i++) {
    temp = a2->blk[i] + 1;
    carryIn = (temp == 0);
    result->blk[i] = temp;
  }
  // If the carry was resolved but the larger number
  // still has blocks, copy them over.
  for (; i < a2->len; i++)
    result->blk[i] = a2->blk[i];
  // Set the extra block if there's still a carry, decrease length otherwise
  if (carryIn)
    result->blk[i] = 1;
  else
    result->len--;
  return 0;
}
inline int subtractBU(BigUnsigned *result, BigUnsigned *a, BigUnsigned *b) {
  BigUnsigned copy;
  if (a == result) {
    initBUBU(&copy, a);
    a = &copy;
  }
  if (b == result) {
    initBUBU(&copy, b);
    b = &copy;
  }
  if (b->len == 0) {
    // If b is zero, copy a.
    return initBUBU(result, a);
  }
  //else if (a->len < b->len)
  // // If a is shorter than b, the result is negative.
  // throw "BigUnsigned::subtract: "
  //   "Negative result in unsigned calculation";
  // Some variables...
  bool borrowIn, borrowOut;
  Blk temp;
  Index i;
  // Set preliminary length and make room
  result->len = a->len;
  // For each block index that is present in both inputs...
  for (i = 0, borrowIn = false; i < b->len; i++) {
    temp = a->blk[i] - b->blk[i];
    // If a reverse rollover occurred,
    // the result is greater than the block from a.
    borrowOut = (temp > a->blk[i]);
    // Handle an incoming borrow
    if (borrowIn) {
      borrowOut |= (temp == 0);
      temp--;
    }
    result->blk[i] = temp; // Save the subtraction result
    borrowIn = borrowOut; // Pass the borrow along
  }
  // If there is a borrow left over, decrease blocks until
  // one does not reverse rollover.
  for (; i < a->len && borrowIn; i++) {
    borrowIn = (a->blk[i] == 0);
    result->blk[i] = a->blk[i] - 1;
  }
  /* If there's still a borrow, the result is negative.
  * Throw an exception, but zero out this object so as to leave it in a
  * predictable state. */
  if (borrowIn) {
    result->len = 0;
    //throw "BigUnsigned::subtract: Negative result in unsigned calculation";
  }
  else
    // Copy over the rest of the blocks
    for (; i < a->len; i++)
      result->blk[i] = a->blk[i];
  // Zap leading zeros
  zapLeadingZeros(result);
  return 0;
}
inline int addIBU(BigUnsigned *result, BigUnsigned *a, Blk b) {
  BigUnsigned temp = { 0 };
  initBlkBU(&temp, b);
  return addBU(result, a, &temp);
}
inline int subtractIBU(BigUnsigned *result, BigUnsigned *a, Blk b) {
  BigUnsigned temp = { 0 };
  initBlkBU(&temp, b);
  return subtractBU(result, a, &temp);
}

//~~BITWISE OPERATORS~~//
/* These are straightforward blockwise operations except that they differ in
* the output length and the necessity of zapLeadingZeros. */
inline int andBU(BigUnsigned *result, BigUnsigned *a, BigUnsigned *b) {
  BigUnsigned copy;
  if (a == result) {
    initBUBU(&copy, a);
    a = &copy;
  }
  if (b == result) {
    initBUBU(&copy, b);
    b = &copy;
  }
  initBU(result);
  // The bitwise & can't be longer than either operand.
  result->len = (a->len >= b->len) ? b->len : a->len;
  for (Index i = 0; i < result->len; i++)
    result->blk[i] = a->blk[i] & b->blk[i];
  //zapLeadingZeros(result);
  return 0;
}
inline int orBU(BigUnsigned *result, BigUnsigned *a, BigUnsigned *b) {
  BigUnsigned copy;
  if (a == result) {
    initBUBU(&copy, a);
    a = &copy;
  }
  if (b == result) {
    initBUBU(&copy, b);
    b = &copy;
  }
  Index i;
  BigUnsigned *a2, *b2;
  if (a->len >= b->len) {
    a2 = a;
    b2 = b;
  }
  else {
    a2 = b;
    b2 = a;
  }
  for (i = 0; i < b2->len; i++)
    result->blk[i] = a2->blk[i] | b2->blk[i];
  for (; i < a2->len; i++)
    result->blk[i] = a2->blk[i];
  result->len = a2->len;
  // Doesn't need zapLeadingZeros.
  return 0;
}
inline int xOrBU(BigUnsigned *result, BigUnsigned *a, BigUnsigned *b) {
  BigUnsigned copy;
  if (a == result) {
    initBUBU(&copy, a);
    a = &copy;
  }
  if (b == result) {
    initBUBU(&copy, b);
    b = &copy;
  }
  Index i;
  BigUnsigned *a2, *b2;
  if (a->len >= b->len) {
    a2 = a;
    b2 = b;
  }
  else {
    a2 = b;
    b2 = a;
  }
  for (i = 0; i < b2->len; i++)
    result->blk[i] = a2->blk[i] ^ b2->blk[i];
  for (; i < a2->len; i++)
    result->blk[i] = a2->blk[i];
  result->len = a2->len;
  zapLeadingZeros(result);
  return 0;
}
inline int shiftBURight(BigUnsigned *result, BigUnsigned *a, int b) {
  BigUnsigned copy;
  if (a == result) {
    initBUBU(&copy, a);
    a = &copy;
  }
  initBU(result);

  // This calculation is wacky, but expressing the shift as a left bit shift
  // within each block lets us use getShiftedBlock.
  Index rightShiftBlocks = (b + numBUBits - 1) / numBUBits;
  int leftShiftBits = numBUBits * rightShiftBlocks - b;
  // Now (N * rightShiftBlocks - leftShiftBits) == b
  // and 0 <= leftShiftBits < N.
  if (rightShiftBlocks >= a->len + 1) {
    // All of a is guaranteed to be shifted off, even considering the left
    // bit shift.
    result->len = 0;
    return 0;
  }
  // Now we're allocating a positive amount.
  // + 1: room for high bits nudged left into another block
  result->len = a->len + 1 - rightShiftBlocks;
  Index i, j;
  for (j = rightShiftBlocks, i = 0; j <= a->len; j++, i++)
    result->blk[i] = getShiftedBUBlock(a, j, leftShiftBits);
  // Zap possible leading zero
  if (result->blk[result->len - 1] == 0)
    result->len--;
  return 0;
}
inline int shiftBULeft(BigUnsigned *result, BigUnsigned *a, unsigned b) {
  BigUnsigned copy;
  if (a == result) {
    initBUBU(&copy, a);
    a = &copy;
  }
  initBU(result);

  Index shiftBlocks = b / numBUBits;
  int shiftBits = b % numBUBits;
  // + 1: room for high bits nudged left into another block
  result->len = a->len + shiftBlocks + 1;
  Index i, j;
  for (i = 0; i < shiftBlocks; i++)
    result->blk[i] = 0;
  for (j = 0, i = shiftBlocks; j <= a->len; j++, i++)
    result->blk[i] = getShiftedBUBlock(a, j, shiftBits);
  // Zap possible leading zero
  if (result->blk[result->len - 1] == 0)
    result->len--;
  return 0;
}
