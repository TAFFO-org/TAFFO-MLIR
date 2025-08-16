// compare_affine_vs_boost.cpp
// Small benchmarks comparing LibAffine Var range propagation with
// Boost interval arithmetic for a few expressions to show strengths
// and weaknesses.

#include "../../include/libaffine.hpp"
#include <boost/numeric/interval.hpp>
#include <iostream>

using namespace LibAffine;
using namespace boost::numeric;

using Interval = interval<double>;

static void print_header(const std::string &title) {
  std::cout << "---- " << title << " ----" << std::endl;
}

// Build simple benchmark cases that highlight over-approximation
// and correlation-awareness of affine forms vs intervals.

static void case1_repeated_use_and_multiply() {
  // Case 2: repeated use and multiplication
  print_header("Case 1: (x - 0.5)*(x - 0.5)");
  Range r2(llvm::APFloat(0.0), llvm::APFloat(1.0));
  Var x2(r2);
  Var y2 = (x2 - llvm::APFloat(0.5)) * (x2 - llvm::APFloat(0.5));
  std::cout << "LibAffine: " << y2.print_affine_form();
  auto yr = y2.get_range();
  std::cout << "Range: " << yr.start.convertToDouble() << " .. "
            << yr.end.convertToDouble() << std::endl;

  Interval xi2(0.0, 1.0);
  Interval yi2 = (xi2 - 0.5) * (xi2 - 0.5);
  std::cout << "Boost.Interval range: " << yi2.lower() << " .. " << yi2.upper()
            << std::endl;
}

static void case2_cross_term_xy() {
  // Case 3: non-linear cross term x*y where x,y in [-1,1]
  print_header("Case 2: x*y with independent vars in [-1,1]");
  Range r3(llvm::APFloat(-1.0), llvm::APFloat(1.0));
  Var xa(r3);
  Var ya(r3);
  Var za = xa * ya;
  std::cout << "LibAffine: " << za.print_affine_form();
  auto zr = za.get_range();
  std::cout << "Range: " << zr.start.convertToDouble() << " .. "
            << zr.end.convertToDouble() << std::endl;

  Interval xii(-1.0, 1.0);
  Interval yii(-1.0, 1.0);
  Interval zii = xii * yii;
  std::cout << "Boost.Interval range: " << zii.lower() << " .. " << zii.upper()
            << std::endl;
}

static void case3_degree10_polynomial() {
  // Case 4: degree-10 polynomial
  print_header("Case 3: degree-10 polynomial on x in [-1,1]");
  Range rx(llvm::APFloat(-1.0), llvm::APFloat(1.0));
  Var x(rx);

  // Coefficients c0 .. c10 (mixed magnitudes to stress over-approximation)
  std::vector<llvm::APFloat> coeffs = {
      llvm::APFloat(0.5),    // c0
      llvm::APFloat(-1.0),   // c1
      llvm::APFloat(0.75),   // c2
      llvm::APFloat(-0.33),  // c3
      llvm::APFloat(2.0),    // c4
      llvm::APFloat(-1.5),   // c5
      llvm::APFloat(0.25),   // c6
      llvm::APFloat(-0.125), // c7
      llvm::APFloat(1.0),    // c8
      llvm::APFloat(-0.5),   // c9
      llvm::APFloat(0.05)    // c10
  };

  // Build powers
  Var x2 = x * x;
  Var x3 = x2 * x;
  Var x4 = x3 * x;
  Var x5 = x4 * x;
  Var x6 = x5 * x;
  Var x7 = x6 * x;
  Var x8 = x7 * x;
  Var x9 = x8 * x;
  Var x10 = x9 * x;

  // Evaluate polynomial p(x) = c0 + c1*x + c2*x^2 + ... + c10*x^10
  Var p = (x * coeffs[1]) + coeffs[0];
  p = p + (x2 * coeffs[2]);
  p = p + (x3 * coeffs[3]);
  p = p + (x4 * coeffs[4]);
  p = p + (x5 * coeffs[5]);
  p = p + (x6 * coeffs[6]);
  p = p + (x7 * coeffs[7]);
  p = p + (x8 * coeffs[8]);
  p = p + (x9 * coeffs[9]);
  p = p + (x10 * coeffs[10]);

  std::cout << "LibAffine polynomial affine form:\n" << p.print_affine_form();
  auto pr = p.get_range();
  std::cout << "Range: " << pr.start.convertToDouble() << " .. "
            << pr.end.convertToDouble() << std::endl;

  // Boost interval evaluation
  Interval xi(-1.0, 1.0);
  std::vector<double> cd = {0.5,  -1.0,   0.75, -0.33, 2.0, -1.5,
                            0.25, -0.125, 1.0,  -0.5,  0.05};
  Interval res = Interval(cd[0]);
  res = res + xi * cd[1];
  res = res + xi * xi * cd[2];
  res = res + xi * xi * xi * cd[3];
  res = res + xi * xi * xi * xi * cd[4];
  res = res + xi * xi * xi * xi * xi * cd[5];
  res = res + xi * xi * xi * xi * xi * xi * cd[6];
  res = res + xi * xi * xi * xi * xi * xi * xi * cd[7];
  res = res + xi * xi * xi * xi * xi * xi * xi * xi * cd[8];
  res = res + xi * xi * xi * xi * xi * xi * xi * xi * xi * cd[9];
  res = res + xi * xi * xi * xi * xi * xi * xi * xi * xi * xi * cd[10];

  std::cout << "Boost.Interval polynomial range: " << res.lower() << " .. "
            << res.upper() << std::endl;
}

static void case4_small_neural_network() {
  // Case 5: very small neural network (2 inputs -> 2 hidden -> 1 output)
  // Inputs x1,x2 in [-1,1]
  print_header("Case 4: tiny 2-2-1 neural network with cubic tanh approx");

  Range rin(llvm::APFloat(-1.0), llvm::APFloat(1.0));
  Var x1(rin);
  Var x2(rin);

  // Define weights and biases (simple values)
  // Hidden layer weights: h1 = w11*x1 + w12*x2 + b1
  //                    h2 = w21*x1 + w22*x2 + b2
  llvm::APFloat w11 = llvm::APFloat(0.8);
  llvm::APFloat w12 = llvm::APFloat(-0.4);
  llvm::APFloat b1 = llvm::APFloat(0.1);

  llvm::APFloat w21 = llvm::APFloat(0.3);
  llvm::APFloat w22 = llvm::APFloat(0.9);
  llvm::APFloat b2 = llvm::APFloat(-0.2);

  // Hidden pre-activations
  Var h1_pre = x1 * w11 + x2 * w12 + b1;
  Var h2_pre = x1 * w21 + x2 * w22 + b2;

  // Activation: use cubic approximation of tanh: tanh(u) ~= u - u^3/3
  auto cubic_tanh = [](const Var &u) {
    Var u3 = u * u * u;
    return u - (u3 / llvm::APFloat(3.0));
  };

  Var h1 = cubic_tanh(h1_pre);
  Var h2 = cubic_tanh(h2_pre);

  // Output layer: y = v1*h1 + v2*h2 + bo
  llvm::APFloat v1 = llvm::APFloat(1.2);
  llvm::APFloat v2 = llvm::APFloat(-0.7);
  llvm::APFloat bo = llvm::APFloat(0.05);

  Var y = h1 * v1 + h2 * v2 + bo;

  std::cout << "LibAffine network output affine form:\n"
            << y.print_affine_form();
  auto yr = y.get_range();
  std::cout << "Range: " << yr.start.convertToDouble() << " .. "
            << yr.end.convertToDouble() << std::endl;

  // Boost interval equivalent
  Interval xi(-1.0, 1.0);
  // hidden pre
  Interval h1p = xi * 0.8 + xi * -0.4 + 0.1;
  Interval h2p = xi * 0.3 + xi * 0.9 + -0.2;
  // cubic tanh approx: u - u^3/3
  auto cubic_tanh_i = [](const Interval &u) { return u - (u * u * u) / 3.0; };
  Interval h1i = cubic_tanh_i(h1p);
  Interval h2i = cubic_tanh_i(h2p);
  Interval yi = h1i * 1.2 + h2i * -0.7 + 0.05;
  std::cout << "Boost.Interval network output range: " << yi.lower() << " .. "
            << yi.upper() << std::endl;
}

// Case 7: (x - y) + (y - x) should be exactly 0 for correlated vars
static void case5_sum_cancel() {
  print_header("Case 5: (x - y) + (y - x) cancellation");
  Range r(llvm::APFloat(0.0), llvm::APFloat(1.0));
  Var x(r);
  Var y(r);
  Var expr = (x - y) + (y - x);
  std::cout << "LibAffine: " << expr.print_affine_form();
  auto er = expr.get_range();
  std::cout << "Range: " << er.start.convertToDouble() << " .. "
            << er.end.convertToDouble() << std::endl;

  Interval xi(0.0, 1.0);
  Interval yi(0.0, 1.0);
  Interval ie = (xi - yi) + (yi - xi);
  std::cout << "Boost.Interval: " << ie.lower() << " .. " << ie.upper()
            << std::endl;
}

// Case 8: (x - x) * y should be 0 but intervals overapproximate
static void case6_zero_times_expr() {
  print_header("Case 6: (x - x) * y (zero times expression)");
  Range r(llvm::APFloat(0.0), llvm::APFloat(1.0));
  Var x(r);
  Var y(r);
  Var expr = (x - x) * y;
  std::cout << "LibAffine: " << expr.print_affine_form();
  auto er = expr.get_range();
  std::cout << "Range: " << er.start.convertToDouble() << " .. "
            << er.end.convertToDouble() << std::endl;

  Interval xi(0.0, 1.0);
  Interval ie = (xi - xi) * xi;
  std::cout << "Boost.Interval: " << ie.lower() << " .. " << ie.upper()
            << std::endl;
}

// Case 9: correlated vs independent variables
static void case7_correlated_vars() {
  print_header("Case 7: correlated vs independent variables");
  Range r(llvm::APFloat(-1.0), llvm::APFloat(1.0));
  Var x(r);

  // independent
  Var y_ind(r);
  Var diff_ind = x - y_ind;
  std::cout << "LibAffine (independent) affine: "
            << diff_ind.print_affine_form();
  auto ri = diff_ind.get_range();
  std::cout << "Range: " << ri.start.convertToDouble() << " .. "
            << ri.end.convertToDouble() << std::endl;

  Interval xi(-1.0, 1.0);
  Interval id_alias = xi - xi; // intervals lose correlation
  std::cout << "Boost.Interval (alias): " << id_alias.lower() << " .. "
            << id_alias.upper() << std::endl;
}

int main() {
  case1_repeated_use_and_multiply();
  case2_cross_term_xy();
  // Case 3: degree-10 polynomial
  // p(x) = c0 + c1*x + c2*x^2 + ... + c10*x^10, computed with Horner
  case3_degree10_polynomial();
  // Case 4: very small neural network
  case4_small_neural_network();
  // Additional cases
  case5_sum_cancel();
  case6_zero_times_expr();
  case7_correlated_vars();

  return 0;
}
