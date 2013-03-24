/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) Scilab Enterprises - 2013 - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/

#include <math.h>
#include "initCudaMatrix.h"

__global__ void initMatrix_kernel(double d, int elems, double* data)
{
    int posInGrid  = blockIdx.x  * blockDim.x + threadIdx.x;

    if(posInGrid < elems)
    {
        data[posInGrid] = d;
    }
}

cudaError_t initCudaMatrix(double h, int iSize, double* d_data)
{
    cudaError_t cudaStat = cudaGetLastError();

    try
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        int threadMax = deviceProp.maxThreadsDim[0];
        int gridMax   = deviceProp.maxGridSize[0];
        int iDimBlock = (iSize > threadMax) ? threadMax : iSize;
        int dimgrid   = (int) ceil((float)iDimBlock/iSize);

        dim3 block(iDimBlock, 1, 1);
        dim3 grid(dimgrid, 1, 1);
        initMatrix_kernel<<<grid, block>>>(h, iSize, d_data);

        iSize -= iDimBlock*dimgrid;
        double* gdblData = d_data;
        while(iSize > 0)
        {
            gdblData += iDimBlock*dimgrid;

            iDimBlock = (iSize > threadMax) ? threadMax : iSize;
            dimgrid   = (int) ceil((float)iDimBlock/iSize);

            dim3 block(iDimBlock, 1, 1);
            dim3 grid(dimgrid, 1, 1);
            initMatrix_kernel<<<grid, block>>>(h, iSize, gdblData);

            iSize -= iDimBlock*dimgrid;
        }

        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        cudaStat = cudaThreadSynchronize();
        if (cudaStat != cudaSuccess) throw cudaStat;

        return cudaSuccess;
    }
    catch(cudaError_t cudaE)
    {
        return cudaE;
    }
}
