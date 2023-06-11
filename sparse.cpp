#include "sparse.hpp"
#include <vector>

void SparseVector::AddEntry(unsigned int i, long double val) {
    this->idxs.push_back(i);
    this->vals.push_back(val);
}

long double SparseVector::Dot(std::vector<long double>& v) {
    long double result = 0;
    for (unsigned int i = 0; i < this->idxs.size(); i++) {
        result += v[this->idxs[i]] * this->vals[i];
    }
    return result;
}

void SparseVector::Update(long double epsilon, long double lambda, long double lr, std::vector<long double>& w) {
    for (unsigned int i = 0; i < this->idxs.size(); i++) {
        w[this->idxs[i]] = w[this->idxs[i]] - 2 * lr * (-epsilon * this->vals[i] + lambda * w[this->idxs[i]]);
    }
}

unsigned long SparseVector::Size() {
    return this->idxs.size();
}