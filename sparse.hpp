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
        void Update(long double epsilon, long double lambda, long double lr, std::vector<long double>& w, unsigned int movieIdx);
        unsigned long Size();
};

#endif /* SPARSE_HPP */