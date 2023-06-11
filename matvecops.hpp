#ifndef MVOPS
#define MVOPS

#include <vector>

// take sum of vector elements
long double sum(std::vector<long double>& v);

// add or subtract two same-length vectors
std::vector<long double> add(
    std::vector<long double>& v1,
    std::vector<long double>& v2,
    bool subtract);

// scalar multiplication between vector and constant
std::vector<long double> scale(std::vector<long double>& v, long double c);

// dot product between two vectors
long double dot(std::vector<long double>& v1, std::vector<long double>& v2);

// print vector to cout, useful for debugging
void print_vec(std::vector<long double>& v);

#endif