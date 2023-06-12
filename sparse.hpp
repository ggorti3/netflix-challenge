#ifndef SPARSE_HPP
#define SPARSE_HPP

#include <vector>

class SparseVector {
    public:
        std::vector<unsigned int> idxs;
        std::vector<long double> vals;
        void AddEntry(unsigned int i, long double val);
        long double Dot(std::vector<long double>& v);
        // long double WComponent(
        //     std::vector<long double>& w,
        //     long double & b,
        //     std::vector<long double>& bu,
        //     std::vector<long double>& bm,
        //     unsigned int userIdx,
        //     unsigned int movieIdx
        // );
        // void Update(
        //     long double epsilon,
        //     long double lambda,
        //     long double lr,
        //     std::vector<long double>& w,
        //     long double & b,
        //     std::vector<long double>& bu,
        //     std::vector<long double>& bm,
        //     unsigned int userIdx,
        //     unsigned int movieIdx
        // );
        unsigned long Size();
};

#endif /* SPARSE_HPP */