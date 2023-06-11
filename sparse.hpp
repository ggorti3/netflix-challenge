#ifndef SPARSE_HPP
#define SPARSE_HPP

#include <vector>

class SparseVector {
    private:
        std::vector<unsigned int> idxs;
        std::vector<long double> vals;
    
    public:
        void AddEntry(unsigned int i, long double val);
        long double Dot(std::vector<long double>& v);
};

#endif /* SPARSE_HPP */