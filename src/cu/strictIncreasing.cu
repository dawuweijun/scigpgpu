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
#include "strictIncreasing.h"

__global__ void strictIncreasing_kernel(double* data, int size, int* result)
{
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if(xIndex >= size-1)
    {
        return;
    }

    if(xIndex == 0)
    {
        *result = 1;
    }
    __syncthreads();

    if(data[xIndex] >= data[xIndex+1])
    {
        *result = 0;
    }
}

cudaError_t cudaStrictIncreasing(double* d_input, int iSize, int* isStrictIncreasing)
{
    try
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        cudaError_t cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        int threadMax   = deviceProp.maxThreadsDim[0];
        int dimblock    = ((iSize-1) > threadMax) ? threadMax : iSize-1;
        int dimgrid     = (int) ceil((float)(iSize-1)/threadMax);

        int* d_result = NULL;
        cudaMalloc((void**)&d_result,sizeof(int));
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        dim3 block(dimblock, 1, 1);
        dim3 grid(dimgrid, 1, 1);

        strictIncreasing_kernel<<<grid, block>>>(d_input, iSize, d_result);

        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        cudaStat = cudaThreadSynchronize();
        if (cudaStat != cudaSuccess) throw cudaStat;

        cudaMemcpy(isStrictIncreasing, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        return cudaSuccess;
    }
    catch(cudaError_t cudaE)
    {
        return cudaE;
    }
}
