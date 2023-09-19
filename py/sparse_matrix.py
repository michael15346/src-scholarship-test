from typing import List

import numpy as np
import scipy

class Element:
    def __init__(self, index, value):
        self.index = index
        self.value = value


class SparseMatrix:
    def __init__(self, *, file=None, dense: np.array = None):
        self.data_: List[List[Element]] = []

        if file is not None:
            assert dense is None
            with open(file, 'r') as f:
                n = int(f.readline())
                for _ in range(n):
                    self.data_.append([])
                    line = f.readline().split()
                    for j in range(1, len(line), 2):
                        self.data_[-1].append(Element(index=int(line[j]),
                                                      value=float(line[j+1])))

        if dense is not None:
            assert file is None
            n = len(dense)
            for i in range(n):
                self.data_.append([])
                for j in range(n):
                    if dense[i, j] != 0:
                        self.data_[i].append(Element(j, dense[i, j]))

    def print(self):
        n = len(self.data_)
        print(f'{len(self.data_)} rows')
        for i in range(n):
            element_index = 0
            for j in range(n):
                if element_index < len(self.data_[i]):
                    if self.data_[i][element_index].index == j:
                        print(f'{self.data_[i][element_index].value:.2f} ', end='')
                    else:
                        print(f'{0:.2f} ', end='')
                else:
                    print(f'{0:.2f} ', end='')
            print()

    def to_dense(self):
        n = len(self.data_)
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            for element in self.data_[i]:
                A[i, element.index] = element.value
        return A

    # This might have been better; but sadly 1.5h isn't nearly enough to write all my ideas into code hehe
    def __matmul__(self, other):

        # there 's like legit a lot of existing fast parallel (gpu whatever) algorithms that work and exist publicly
        # it will be easier to just use those
        # vvvvv your code here vvvvv


        # Now for some ideas:
        # you can easily parallelize across m by n workers if you dont care about memory and multiple computations.
        # you can transform the "custom form" of the matrix if they have  many dense blocks; then multiply them with usual block matrix thing
        # once again, you could use binary & to find indices that are in both matrices; then only calculate those.
        n_left = len(self.data_)
        n_right = len(other.data_)
        sparse_left = scipy.sparse.lil_matrix((n_left,n_left), dtype=float)
        sparse_right = scipy.sparse.lil_matrix((n_right, n_right), dtype=float)
        for i in range(len(self.data_)):
            for element in self.data_[i]:
                sparse_left[i, element.index] = element.value

        for i in range(len(other.data_)):
            for element in other.data_[i]:
                sparse_right[i, element.index] = element.value
        sparse_mul = sparse_left @ sparse_right
        result = SparseMatrix(dense=sparse_mul.todense())
        # ^^^^^ your code here ^^^^^

        return result


    def __pow__(self, power, modulo=None):

        # vvvvv your code here vvvvv
        # there might be a cute algorithm to do this efficiently, but the stupid idea is to do matmul until we're done
        # it might be faster to use eig decomposition, but idk honestly, seems a bit too expensive for really sparse matrices
        # or something else even

        # Some more ideas:
        # a) use trivial algorithm, but generate once the list of indices to calculate, then do the usual o(log n) power using matmul
        # b) if there's a fast way to calculate the eigenvalues, then you can use the fact that A^k = V * D^k * V^-1
        # where D is a diagonal matrix of eigenvalues and V is a matrix of eigenvectors
        # c) i think numpy power does the o (log n) algorithm anyway (at least it should for integers)
        # d): considering a), if the matrix is sufficiently sparse, there will only be a handful of indices to calculate


        n = len(self.data_)
        sparse = scipy.sparse.lil_matrix((n,n), dtype=float)
        for i in range(len(self.data_)):
            for element in self.data_[i]:
                sparse[i, element.index] = element.value


        result = SparseMatrix(dense=(sparse ** power).todense())
        # ^^^^^ your code here ^^^^^

        return result
