#ifndef ODOMETRY_TEST_HELPERS_H_
#define ODOMETRY_TEST_HELPERS_H_

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#include <Eigen/Dense>
#include <functional>

namespace test_helpers {

Eigen::MatrixXd der_check(
    const Eigen::VectorXd& der_at,
    const std::function< Eigen::VectorXd(const Eigen::VectorXd&) >& numeric,
    const std::function< Eigen::MatrixXd(const Eigen::VectorXd&) >& analytic,
    bool print = false);

std::vector<int> buildRange(int end);

}

const static Eigen::IOFormat csvFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

namespace {
// Useful for creating unit tests (?)
inline void write_csv(const std::string& path, const Eigen::MatrixXd& A) {
    std::ofstream file(path.c_str());
    file << A.format(csvFormat);
}

template<typename M>
M load_csv(const std::string& path) {
    std::ifstream indata;
    indata.open(path);
    if (!indata.is_open()) {
        std::cout << "Could not open file " << path << std::endl;
    }
    assert(indata.is_open());

    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    assert(rows > 0);
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
}

}

#endif
