#ifndef SPARSE_HPP
#define SPARSE_HPP

#include <vector>

class SparseVector {
    private:
        std::vector<unsigned int> idxs;
        std::vector<double> vals;
    
    public:
        void AddEntry(unsigned int i, double val);
        double Dot(std::vector<double>& v);
};

#endif /* SPARSE_HPP */