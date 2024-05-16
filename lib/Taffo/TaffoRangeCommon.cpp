#include "Taffo/TaffoRangeCommon.h"
#include "llvm/ADT/APFloat.h"

using namespace ::llvm;

namespace taffo {
  template<>
  std::pair<APFloat, APFloat> inferAssign(ArrayRef<std::pair<APFloat, APFloat>> argRanges) {
    assert(argRanges[0].first <= argRanges[0].second);
    return argRanges[0];
  }


  template<>
  std::pair<APFloat, APFloat> inferAdd(ArrayRef<std::pair<APFloat, APFloat>> argRanges) {
    assert(argRanges[0].first <= argRanges[0].second);
    assert(argRanges[1].first <= argRanges[1].second);

    APFloat min = argRanges[0].first + argRanges[1].first;
    APFloat max = argRanges[0].second + argRanges[1].second;

    return std::pair<APFloat, APFloat>(min, max);
  }

  template<>
  std::pair<APFloat, APFloat> inferMult(ArrayRef<std::pair<APFloat, APFloat>> argRanges) {
    // unrolled outer product
    APFloat a0b0 = argRanges[0].first * argRanges[1].first;
    APFloat a0b1 = argRanges[0].first * argRanges[1].second;
    APFloat a1b0 = argRanges[0].second * argRanges[1].first;
    APFloat a1b1 = argRanges[0].second * argRanges[1].second;

    return std::minmax({a0b0, a0b1, a1b0, a1b1});
  }
}