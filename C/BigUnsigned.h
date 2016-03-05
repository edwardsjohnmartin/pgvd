#ifndef BIGUNSIGNED_C
#define BIGUNSIGNED_C
#define BIG_INTEGER_SIZE 10

typedef unsigned long Index; // Type for the index of a block in the array
typedef unsigned long Blk;  // Type for the blocks

// BigUnsigned allows storing integers larger than a long using an array of blk.
typedef struct {
	Index len;                                      // Actual length of the value stored (in blocks)
	bool isNULL;
	Blk blk[BIG_INTEGER_SIZE];
} BigUnsigned;

//~~HELPER FUNCTIONS~~//
// Decreases len to eliminate any leading zero blocks.
void zapLeadingZeros(BigUnsigned * bu);
int isBUZero(BigUnsigned *bu);
void printBUSize();

//~~INITIALIZERS~~//
int initBUBU(BigUnsigned *result, BigUnsigned *x);
int initBU(BigUnsigned *result);
int initBlkBU(BigUnsigned *result, Blk x);
//ASSUMES sizeof Blk = sizeof unsigned char!!!
int initLongLongBU(BigUnsigned *result, long long x);
int initMorton(BigUnsigned *result, Blk x);
int initNULLBU(BigUnsigned *result);

//~~BIT/BLOCK ACCESSORS~~//
Blk getBUBlock(BigUnsigned *bu, Index i);
void setBUBlock(BigUnsigned *bu, Index i, Blk newBlock);
Blk getShiftedBUBlock(BigUnsigned *num, Index x, unsigned int y);
Index getBUBitLength(BigUnsigned *bu);
bool getBUBit(BigUnsigned *bu, Index bi);
void setBUBit(BigUnsigned *bu, Index bi, bool newBit);

//~~COMPARISON~~//
int compareBU(BigUnsigned *x, BigUnsigned *y);

//~~ARITHMATIC OPERATIONS~~//
int addBU(BigUnsigned *result, BigUnsigned *a, BigUnsigned *b);
int subtractBU(BigUnsigned *result, BigUnsigned *a, BigUnsigned *b);
int addIBU(BigUnsigned *result, BigUnsigned *a, Blk b);
int subtractIBU(BigUnsigned *result, BigUnsigned *a, Blk b);


//~~BITWISE OPERATORS~~//
/* These are straightforward blockwise operations except that they differ in
* the output length and the necessity of zapLeadingZeros. */
int andBU(BigUnsigned *result, BigUnsigned *a, BigUnsigned *b);
int orBU(BigUnsigned *result, BigUnsigned *a, BigUnsigned *b);
int xOrBU(BigUnsigned *result, BigUnsigned *a, BigUnsigned *b);
int shiftBURight(BigUnsigned *result, BigUnsigned *a, int b);
int shiftBULeft(BigUnsigned *result, BigUnsigned *a, int b);

#endif
