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
#include "extract.h"

__device__ int d_iErrExtract;
__global__ void initErrExtract()
{
    d_iErrExtract = 0;
}

__global__ void extract_kernel(double* d_inputA, int inputSize, double* d_output, double* pdblPos, int outputSize)
{
    int posInGrid = blockIdx.x * blockDim.x + threadIdx.x;

    if(posInGrid < outputSize)
    {
        int extractPos = (int)pdblPos[posInGrid];
        if(extractPos <= 0 || extractPos > inputSize)
        {
            d_iErrExtract=-1;
            return;
        }

        d_output[posInGrid] = d_inputA[extractPos - 1];
    }
}

__global__ void extractZ_kernel(cuDoubleComplex* d_inputA, int inputSize, cuDoubleComplex* d_output, double* pdblPos, int outputSize)
{
    int posInGrid = blockIdx.x * blockDim.x + threadIdx.x;

    if(posInGrid < outputSize)
    {
        int extractPos = (int)pdblPos[posInGrid];
        if(extractPos <= 0 || extractPos > inputSize)
        {
            d_iErrExtract=-1;
            return;
        }

        d_output[posInGrid].x = d_inputA[extractPos - 1].x;
        d_output[posInGrid].y = d_inputA[extractPos - 1].y;
    }
}

cudaError_t cudaExtract(double* d_inputA, int inputSize, double* d_output, double* pdblPos, int outputSize, int* piErr)
{
	cudaError_t cudaStat = cudaGetLastError();
    *piErr = 0;

	try
	{
	    // get device properties
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess) throw cudaStat;

		int threadMax = deviceProp.maxThreadsDim[0];
		int dimgrid   = (int)ceil((float)outputSize/threadMax);

        // perform operation
        dim3 block(threadMax, 1, 1);
        dim3 grid(dimgrid, 1, 1);
        initErrExtract<<<1,1>>>();
        extract_kernel<<<grid, block>>>(d_inputA, inputSize, d_output, pdblPos, outputSize);

        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        // get error flag from device
        cudaMemcpyFromSymbol(piErr, "d_iErrExtract", sizeof(int), 0, cudaMemcpyDeviceToHost);
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

cudaError_t cudaZExtract(cuDoubleComplex* d_inputA, int inputSize, cuDoubleComplex* d_output, double* pdblPos, int outputSize, int* piErr)
{
	cudaError_t cudaStat = cudaGetLastError();
    *piErr = 0;

	try
	{
	    // get device properties
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess) throw cudaStat;

		int threadMax = deviceProp.maxThreadsDim[0];
		int dimgrid   = (int)ceil((float)outputSize/threadMax);

        // perform operation
        dim3 block(threadMax, 1, 1);
        dim3 grid(dimgrid, 1, 1);
        initErrExtract<<<1,1>>>();
        extractZ_kernel<<<grid, block>>>(d_inputA, inputSize, d_output, pdblPos, outputSize);

        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        // get error flag from device
        cudaMemcpyFromSymbol(piErr, "d_iErrExtract", sizeof(int), 0, cudaMemcpyDeviceToHost);
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
