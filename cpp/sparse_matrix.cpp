#include "sparse_matrix.h"
#include <future>

SparseMatrix::SparseMatrix(fs::path fname) {
    std::ifstream stream(fname);
    int32_t n_rows;
    stream >> n_rows;
    data.resize(n_rows);

    for (int32_t i = 0; i < n_rows; i++) {
        int32_t n_cols;
        stream >> n_cols;
        data[i].reserve(n_cols);
        for (int32_t j = 0; j < n_cols; j++) {
            int32_t index;
            double value;
            stream >> index >> value;
            data[i].push_back({index, value});
        }
    }
}

void SparseMatrix::Print() {
    int32_t n = data.size();
    printf("%d rows:\n", n);
    for (int32_t i = 0; i < n; i++) {
        int32_t element_index = 0;
        for (int32_t j = 0; j < n; j++) {
            if (element_index < data[i].size()) {
                if (data[i][element_index].index == j) {
                    printf("%.2f ", data[i][element_index].value);
                    element_index++;
                } else {
                    printf("%.2f ", 0.);
                }
            } else {
                printf("%.2f ", 0.);
            }
        }
        printf("\n");
    }
}
//Hope you don't mind separate functions.
void matmul_worker(const SparseMatrix& left, const SparseMatrix& right, const SparseMatrix& result) {
    
}

SparseMatrix SparseMatrix::operator*(const SparseMatrix &m) {
    SparseMatrix result;
    //see py for impl that's faster than anything id be able to write here
    return result;
};

SparseMatrix SparseMatrix::operator^(uint32_t p) {
    SparseMatrix result;
    return result;
}