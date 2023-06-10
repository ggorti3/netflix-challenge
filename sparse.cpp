#include "sparse.hpp"
#include "COO2CSR.hpp"
#include "matvecops.hpp"
#include <vector>

void SparseMatrix::Resize(int nrows, int ncols) {
    this->nrows = nrows;
    this->ncols = ncols;
}

void SparseMatrix::AddEntry(int i, int j, double val) {
    this->i_idx.push_back(i);
    this->j_idx.push_back(j);
    this->a.push_back(val);
}

void SparseMatrix::ConvertToCSR() {
    COO2CSR(this->a, this->i_idx, this->j_idx);
}

std::vector<double> SparseMatrix::MulVec(std::vector<double> &vec) {
    return rMult(
        this->a,
        this->j_idx,
        this->i_idx,
        vec
    );
}