#include "libaffine.hpp"

namespace LibAffine {
// Implementing the case where the scalar is on the left side of the operator
Var operator+(const llvm::APFloat b, const Var var) { return var + b; }

Var operator-(const llvm::APFloat b, const Var var) { return -var + b; }

Var operator*(const llvm::APFloat b, const Var var) { return var * b; }

bool operator==(const Var &a, const Var &b) {
  if (a.c_value.isNaN() && b.c_value.isNaN())
    return true;

  llvm::APFloat epsilon(llvm::APFloat::IEEEdouble(), "1e-6");
  llvm::APFloat diff_start = a.get_range().start - b.get_range().start;
  llvm::APFloat diff_end = a.get_range().end - b.get_range().end;
  if (diff_start.compare(epsilon) <= llvm::APFloat::cmpEqual &&
      diff_end.compare(epsilon) <= llvm::APFloat::cmpEqual)
    return true;

  return false;
}

bool operator!=(const Var &a, const Var &b) { return !(a == b); }
} // namespace LibAffine