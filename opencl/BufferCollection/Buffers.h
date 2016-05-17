#pragma once
#include <memory>
#include "buffer.h"
using namespace std;
struct Buffers
{
  shared_ptr<Buffer> points;
  shared_ptr<Buffer> bigUnsignedInput;
  shared_ptr<Buffer> predicate;
  shared_ptr<Buffer> address;
  shared_ptr<Buffer> intermediate;
  shared_ptr<Buffer> bigUnsignedResult;
  shared_ptr<Buffer> internalNodes;
  shared_ptr<Buffer> leafNodes;
  shared_ptr<Buffer> localSplits;
  shared_ptr<Buffer> scannedSplits;
  shared_ptr<Buffer> octree;
};

