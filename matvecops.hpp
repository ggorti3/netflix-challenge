#ifndef MVOPS
#define MVOPS

#include <vector>

// take sum of vector elements
double sum(std::vector<double>& v);

// add or subtract two same-length vectors
std::vector<double> add(
    std::vector<double>& v1,
    std::vector<double>& v2,
    bool subtract);

// scalar multiplication between vector and constant
std::vector<double> scale(std::vector<double>& v, double c);

// dot product between two vectors
double dot(std::vector<double>& v1, std::vector<double>& v2);

// print vector to cout, useful for debugging
void print_vec(std::vector<double>& v);

#endif