#ifndef LIBAFFINE_HPP
#define LIBAFFINE_HPP

#include <stdint.h>

#include "llvm/ADT/APFloat.h"
#include <algorithm>
#include <atomic>
#include <vector>

namespace LibAffine {
#define MAX_ERR_SYMBOLS 1024
class Range {
public:
  llvm::APFloat start;
  llvm::APFloat end;
  Range(llvm::APFloat start, llvm::APFloat end) : start(start), end(end){};
  llvm::APFloat get_central_value() {
    return (start + end) / (llvm::APFloat)2.0;
  };
  llvm::APFloat get_radius() { return (end - start) / (llvm::APFloat)2.0; };
};

class Var {
private:
  Var() {
    err_symbol_coeffs.reserve(MAX_ERR_SYMBOLS);
    err_symbol_index.reserve(MAX_ERR_SYMBOLS);
  };

  inline static std::atomic<unsigned int> highest_err_symbol{0};
  inline static std::atomic<unsigned int> inc_err_symbol_index() {
    return highest_err_symbol++;
  };
  llvm::APFloat c_value = llvm::APFloat(0.0);   // Central value
  std::vector<llvm::APFloat> err_symbol_coeffs; // error symbol coefficients
  std::vector<unsigned int> err_symbol_index;   // error symbol coefficients

public:
  Var(Range range) {
    Var();
    c_value = range.get_central_value();
    err_symbol_coeffs.push_back(range.get_radius());
    err_symbol_index.push_back(inc_err_symbol_index());
  };

  Range get_range() const {
    llvm::APFloat radius = abs_coeff_sum();
    return Range(c_value - radius, c_value + radius);
  }
  llvm::APFloat abs_coeff_sum() const {
    llvm::APFloat sum = llvm::APFloat(0.0);
    for (auto i : err_symbol_coeffs) {
      sum = sum + llvm::abs(i);
    }
    return sum;
  }

  // define the join operation on two affine variables
  Var join(const Var &b) const {
    // assert the central value is the same and the error symbols that are
    // shared have the same coefficient
    Var result;
    result.c_value = (c_value + b.c_value) / (llvm::APFloat)2.0;
    std::set_union(err_symbol_index.begin(), err_symbol_index.end(),
                   b.err_symbol_index.begin(), b.err_symbol_index.end(),
                   std::back_inserter(result.err_symbol_index));

    unsigned int index_ida = 0;
    unsigned int index_idb = 0;
    for (auto i : result.err_symbol_index) {
      if (index_ida >= err_symbol_index.size() ||
          err_symbol_index[index_ida] != i) {
        result.err_symbol_coeffs.push_back(b.err_symbol_coeffs[index_idb]);
        index_idb++;
        continue;
      }
      if (index_idb >= b.err_symbol_index.size() ||
          b.err_symbol_index[index_idb] != i) {
        result.err_symbol_coeffs.push_back(err_symbol_coeffs[index_ida]);
        index_ida++;
        continue;
      }
      result.err_symbol_coeffs.push_back(llvm::maximum(
          err_symbol_coeffs[index_ida], b.err_symbol_coeffs[index_idb]));
      index_ida++;
      index_idb++;
    }
    // Calulate the approximation error in a new symbol
    result.err_symbol_index.push_back(inc_err_symbol_index());
    result.err_symbol_coeffs.push_back(llvm::abs(c_value - b.c_value) /
                                       (llvm::APFloat)2.0);
    return result;
  }

  Var operator+(const Var &b) const { // Addition
    Var result;
    result.c_value = c_value + b.c_value;
    std::set_union(err_symbol_index.begin(), err_symbol_index.end(),
                   b.err_symbol_index.begin(), b.err_symbol_index.end(),
                   std::back_inserter(result.err_symbol_index));

    unsigned int index_ida = 0;
    unsigned int index_idb = 0;
    for (auto i : result.err_symbol_index) {
      if (index_ida >= err_symbol_index.size() ||
          err_symbol_index[index_ida] != i) {
        result.err_symbol_coeffs.push_back(b.err_symbol_coeffs[index_idb]);
        index_idb++;
        continue;
      }
      if (index_idb >= b.err_symbol_index.size() ||
          b.err_symbol_index[index_idb] != i) {
        result.err_symbol_coeffs.push_back(err_symbol_coeffs[index_ida]);
        index_ida++;
        continue;
      }
      result.err_symbol_coeffs.push_back(err_symbol_coeffs[index_ida] +
                                         b.err_symbol_coeffs[index_idb]);
      index_ida++;
      index_idb++;
    }
    return result;
  }

  Var operator+(const llvm::APFloat b) const { // Addition with scalar
    Var result;
    result.c_value = c_value + b;
    for (auto i : err_symbol_coeffs)
      result.err_symbol_coeffs.push_back(i);
    return result;
  }

  Var operator-(const Var &b) const {
    Var result;
    result.c_value = c_value - b.c_value;
    std::set_union(err_symbol_index.begin(), err_symbol_index.end(),
                   b.err_symbol_index.begin(), b.err_symbol_index.end(),
                   std::back_inserter(result.err_symbol_index));

    unsigned int index_ida = 0;
    unsigned int index_idb = 0;
    for (auto i : result.err_symbol_index) {
      if (index_ida >= err_symbol_index.size() ||
          err_symbol_index[index_ida] != i) {
        result.err_symbol_coeffs.push_back(-b.err_symbol_coeffs[index_idb]);
        index_idb++;
        continue;
      }
      if (index_idb >= b.err_symbol_index.size() ||
          b.err_symbol_index[index_idb] != i) {
        result.err_symbol_coeffs.push_back(err_symbol_coeffs[index_ida]);
        index_ida++;
        continue;
      }
      result.err_symbol_coeffs.push_back(err_symbol_coeffs[index_ida] -
                                         b.err_symbol_coeffs[index_idb]);
      index_ida++;
      index_idb++;
    }
    return result;
  }

  Var operator-(const llvm::APFloat b) const {
    Var result;
    result.c_value = c_value - b;
    for (auto i : err_symbol_coeffs)
      result.err_symbol_coeffs.push_back(i);
    return result;
  }

  Var operator-() const {
    Var result;
    result.c_value = -c_value;
    for (auto i : err_symbol_coeffs)
      result.err_symbol_coeffs.push_back(-i);
    return result;
  }

  Var operator*(const Var &b) const { // Approximation of multiplication range
    Var result;
    result.c_value = c_value * b.c_value;
    std::set_union(err_symbol_index.begin(), err_symbol_index.end(),
                   b.err_symbol_index.begin(), b.err_symbol_index.end(),
                   std::back_inserter(result.err_symbol_index));

    unsigned int index_ida = 0;
    unsigned int index_idb = 0;
    for (auto i : result.err_symbol_index) {
      if (index_ida >= err_symbol_index.size() ||
          err_symbol_index[index_ida] != i) {
        result.err_symbol_coeffs.push_back(c_value *
                                           b.err_symbol_coeffs[index_idb]);
        index_idb++;
        continue;
      }
      if (index_idb >= b.err_symbol_index.size() ||
          b.err_symbol_index[index_idb] != i) {
        result.err_symbol_coeffs.push_back(b.c_value *
                                           err_symbol_coeffs[index_ida]);
        index_ida++;
        continue;
      }
      result.err_symbol_coeffs.push_back(
          c_value * b.err_symbol_coeffs[index_idb] +
          b.c_value * err_symbol_coeffs[index_ida]);
      index_ida++;
      index_idb++;
    }

    // Compute the approximation error in a new symbol
    result.err_symbol_index.push_back(inc_err_symbol_index());
    result.err_symbol_coeffs.back() = abs_coeff_sum() * b.abs_coeff_sum();
    return result;
  }

  Var operator*(const llvm::APFloat b) const {
    Var result;
    result.c_value = c_value * b;
    for (auto i : err_symbol_coeffs)
      result.err_symbol_coeffs.push_back(i * b);
    return result;
  }

  // Make == operator a friend function of this class
  friend bool operator==(const Var &a, const Var &b);
};

// Implementing the case where the scalar is on the left side of the operator
Var operator+(const llvm::APFloat b, const Var var) { return var + b; }

Var operator-(const llvm::APFloat b, const Var var) { return -var + b; }

Var operator*(const llvm::APFloat b, const Var var) { return var * b; }

bool operator==(const Var &a, const Var &b) {
  if (a.c_value != b.c_value)
    return false;
  if (a.err_symbol_index != b.err_symbol_index)
    return false;
  return std::equal(a.err_symbol_coeffs.begin(), a.err_symbol_coeffs.end(),
                    b.err_symbol_coeffs.begin());
}

bool operator!=(const Var &a, const Var &b) { return !(a == b); }
} // namespace LibAffine
#endif // LIBAFFINE_HPP