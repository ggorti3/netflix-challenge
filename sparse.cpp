#include "sparse.hpp"
#include <vector>

void SparseVector::AddEntry(unsigned int i, double val) {
    this->idxs.push_back(i);
    this->vals.push_back(val);
}

double SparseVector::Dot(std::vector<double>& v) {
    double result = 0;
    for (unsigned int i = 0; i < this->idxs.size(); i++) {
        result += v[this->idxs[i]] * this->vals[i];
    }
    return result;
}