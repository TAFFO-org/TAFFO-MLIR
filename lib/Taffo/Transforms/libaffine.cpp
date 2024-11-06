#include "libaffine.hpp"

namespace LibAffine {
// Implementing the case where the scalar is on the left side of the operator
Var operator+(const llvm::APFloat b, const Var var) { return var + b; }

Var operator-(const llvm::APFloat b, const Var var) { return -var + b; }

Var operator*(const llvm::APFloat b, const Var var) { return var * b; }

bool operator==(const Var &a, const Var &b) {
  if (a.c_value.isNaN() && b.c_value.isNaN())
    return true;
  // if (a.c_value != b.c_value)
  //   return false;
  // if (a.err_symbol_index != b.err_symbol_index)
  //   return false;
  // return std::equal(a.err_symbol_coeffs.begin(), a.err_symbol_coeffs.end(),
  //                   b.err_symbol_coeffs.begin());
  if (a.get_range().start == b.get_range().start &&
      a.get_range().end == b.get_range().end)
    return true;
  return false;
}

bool operator!=(const Var &a, const Var &b) { return !(a == b); }
} // namespace LibAffine