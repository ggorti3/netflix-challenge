#include <iostream>
#include <vector>
#include "matvecops.hpp"

void print_vec(std::vector<double>& v) {
    
    // set formatting flags
    std::cout.setf(std::ios_base::scientific);
    std::cout.precision(4);

    // iterate through the vector and print each value
    for (uint j = 0; j < (uint)v.size(); j++) {
        std::cout.width(11);
        std::cout << v[j] << std::endl;
    }
}

double sum(std::vector<double>& v) {
    double s = 0;
    // iterate through the vector and print each value
    for (uint j = 0; j < (uint)v.size(); j++) {
        s += v[j];
    }
    return s;
}

std::vector<double> add(
    std::vector<double>& v1,
    std::vector<double>& v2,
    bool subtract) {
    std::vector<double> v;

    // verify that v1 and v2 are same length
    if (v1.size() != v2.size()) {
        std::cout << "dot product: vectors must be same length" << std::endl;
        exit(0);
    }

    // append value to v depending on whether adding or subtracting
    for (unsigned int i = 0; i < v1.size(); i++) {
        if (!subtract) {
            v.push_back(v1[i] + v2[i]);
        } else {
            v.push_back(v1[i] - v2[i]);
        }
    }
    return v;

}

std::vector<double> scale(std::vector<double>& v, double c) {
    std::vector<double> v2;
    // iterate through vector and multiply by scale
    for (unsigned int i = 0; i < v.size(); i++) {
        v2.push_back(c * v[i]);
    }
    return v2;
}

double dot(std::vector<double>& v1, std::vector<double>& v2) {
    // verify that v1 and v2 are same length
    if (v1.size() != v2.size()) {
        std::cout << "dot product: vectors must be same length" << std::endl;
        exit(0);
    }

    // iterate through vectors
    double result = 0;
    for (unsigned int i = 0; i < v1.size(); i++) {
        // increment result by product of entries
        result += v1[i] * v2[i];
    }
    return result;
}