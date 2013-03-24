#include <stdio.h>
#include <stdlib.h>
#include "cublas.h"
#include <algorithm>

cublasStatus decomposeLU(int M, int N, int lda , double* A, int* P);
cublasStatus decomposeBlockedLU(int M, int N, int lda, double *A, int blockSize);
