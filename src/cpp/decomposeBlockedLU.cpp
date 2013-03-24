#include "LU.h"

cublasStatus decomposeBlockedLU(int rows, int cols,int lda, double *dA, int blockSize)
{
    int minSize = std::min(rows,cols);
    int* P = (int*)malloc(lda * sizeof(int));
    cublasStatus cuStat;

    if(blockSize > minSize || blockSize == 1)
    {
        //straight LU decomposition
        cuStat = decomposeLU( rows, cols, lda, dA, P);
        if(cuStat != CUBLAS_STATUS_SUCCESS)
            return cuStat;
    }
    else
    {
        //blocked decomposition
        for(int i =0; i< minSize ; i+=blockSize)
        {
            int realBlockSize  = std::min(minSize - i, blockSize);

            //decompose the current rectangular block
            cuStat = decomposeLU(rows-i, realBlockSize, lda, dA+i+i*lda, P+i);
            if(cuStat != CUBLAS_STATUS_SUCCESS)
                return cuStat;
            //adjust pivot infos
            //Todo : write a kernel for that
            for(int p = i; p < std::min(rows, i+realBlockSize)-1; p++)
            {
                P[p] = P[p]+i;
                if(P[p] != p)
                {
                // Apply interchanges to columns 0:i.
                cublasDswap(i, dA+p , lda, dA+ P[p], lda);
                // Apply interchanges to columns i+blockSize:cols.
                cublasDswap(cols-i-realBlockSize, dA+p+(i+realBlockSize)*lda , lda, dA+ P[p]+(i+realBlockSize)*lda, lda);
                }
            }

            // Compute block row of U.
            cublasDtrsm('l','l','n','u', realBlockSize, cols-i-realBlockSize, 1.0f, dA +i +i*lda, lda, dA +i + (i+realBlockSize)*lda, lda);
            if(i+realBlockSize < rows)
            {
                cublasDgemm(   'n','n',  cols-i-realBlockSize, cols-i-realBlockSize, realBlockSize,
                                        -1.0f,
                                        dA+i+realBlockSize+i*lda,lda,
                                        dA+i+(realBlockSize+i)*lda,lda,
                                        1.0f,
                                        dA+i+realBlockSize+(realBlockSize+i)*lda,lda);
            }
        }
    }
    return cuStat;
}

