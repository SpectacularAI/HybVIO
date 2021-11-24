#include "helpers.hpp"

namespace test_helpers {

// Compare analytic and numerical derivatives of function: \R^n \to \R^m
// at single point \in \R^n.
// Inspired by the matlab function `der_check` by Simo Särkkä.
//
// NOTE I found the code somewhat fragile, not recommending for usage in production.
// In particular, I discovered that if either of the function parameters is defined as
// lambda with no explicit return type, an issue can occur where top 2-3 rows of the
// result derivative difference matrix are garbage. I think the problem is that
// an Eigen block expression returned from the lambda is not compatible with the plain
// VectorXd and MatrixXd types given in the signature here [1], but that this does not
// trigger a compiler warning because there exists conversion from Block to Vector/Matrix.
// My understanding is that Eigen has a couple of issues like this where the dev team
// found no way to trigger compiler warnings and they have to trust the library user to
// do the right thing. I tried using Eigen::Ref<> and Eigen::MatrixBase<> template, but I
// couldn't figure if they can be used in the return value position.
//
// tl;dr  Declare the the two function arguments like this:
//     auto numeric = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd { ... }
//     auto analytic = [&](const Eigen::VectorXd& x) -> Eigen::MatrixXd { ... }
//
// Then any block expressions will be converted into plain matrices already in the lambda
// body. See usage examples in the unit tests.
//
// [1]: <https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html>
//
// `der_at`:   The point (m) to evaluate the functions.
// `numeric`:  n -> m function. Will be evaluated n + 1 times (watch out for reuse of variables).
// `analytic`: function to compute (m * n) matrix of partial derivatives of the `numeric`
//             function at `der_at` (m). Will be evaluated once.
Eigen::MatrixXd der_check(
    const Eigen::VectorXd& der_at,
    const std::function< Eigen::VectorXd(const Eigen::VectorXd&) >& numeric,
    const std::function< Eigen::MatrixXd(const Eigen::VectorXd&) >& analytic,
    bool print
) {
    // Small enough to approximate derivative, but not too small to cause wild inaccuracy
    // with floating point math.
    double h = 1e-7;

    Eigen::MatrixXd der_analytic = analytic(der_at);
    int m = der_analytic.rows();
    int n = der_analytic.cols();

    Eigen::MatrixXd der_numeric = Eigen::MatrixXd::Zero(m, n);
    Eigen::VectorXd y0 = numeric(der_at);
    assert(y0.size() == m);
    for (int i = 0; i < n; i++) {
        Eigen::VectorXd x = der_at;
        x(i) += h;
        der_numeric.col(i) = (numeric(x) - y0) / h;
    }
    if (print) {
        std::cout << "Analytic:" << std::endl;
        std::cout << der_analytic << std::endl;
        std::cout << "Numeric:" << std::endl;
        std::cout << der_numeric << std::endl;
    }
    return (der_analytic - der_numeric).array().abs().matrix();
}

std::vector<int> buildRange(int end) {
    std::vector<int> vec;
    for (int i = 0; i < end; ++i) {
        vec.push_back(i);
    }
    return vec;
}

}
