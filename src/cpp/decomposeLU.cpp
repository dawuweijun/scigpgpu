#include "LU.h"

cublasStatus decomposeLU(int rows, int cols, int lda , double* A, int* P)
{
    cublasStatus cuStat;
    int minDim = std::min(rows, cols);

    for(int k=0; k<minDim-1; k++)
    {
        int pivotRow = k-1+cublasIdamax(rows-k,A+k + k*lda, 1); // row relative to the current submatrix
        int kp1 = k+1;
        P[k] = pivotRow;
        double valcheck;

        if(pivotRow!=k)
        {
            cublasDswap(cols, A+pivotRow, lda, A+k, lda);
        }
        
        cuStat = cublasGetVector(1,sizeof(double),A+k+ k*lda, 1, &valcheck, 1);
        if(cuStat != CUBLAS_STATUS_SUCCESS)
            return cuStat;

        if(kp1 < rows)
        {
            cublasDscal(rows-kp1, 1.0f/valcheck,A+kp1+ k*lda, 1);
        }
        if(kp1 < minDim)
        {
            cublasDger(rows-kp1, cols-kp1, -1.0f,A+kp1+ k*lda, 1, A+k+ kp1*lda, lda,A+ kp1*lda+kp1, lda);
        }
    }
    return cuStat;
}

