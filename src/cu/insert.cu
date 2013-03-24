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
#include "insert.h"

__device__ int d_iErrInsert;
__global__ void initErrInsert()
{
    d_iErrInsert = 0;
}

__global__ void insert_kernel(double* d_inputA, int inputSize, double* d_data, double* pdblPos, int dataSize, int isScalar)
{
    int posInGrid = blockIdx.x * blockDim.x + threadIdx.x;

    if(posInGrid < dataSize)
    {
        int insertPos = (int)pdblPos[posInGrid];
        if(insertPos <= 0 || insertPos > inputSize)
        {
            d_iErrInsert=-1;
            return;
        }

        if(isScalar)
        {
            d_inputA[insertPos - 1] = d_data[0];
        }
        else
        {
            d_inputA[insertPos - 1] = d_data[posInGrid];
        }
    }
}

__global__ void insertZ_kernel(cuDoubleComplex* d_inputA, int inputSize, cuDoubleComplex* d_data, double* pdblPos, int dataSize, int isScalar)
{
    int posInGrid = blockIdx.x * blockDim.x + threadIdx.x;

    if(posInGrid < dataSize)
    {
        int insertPos = (int)pdblPos[posInGrid];
        if(insertPos <= 0 || insertPos > inputSize)
        {
            d_iErrInsert=-1;
            return;
        }

        if(isScalar)
        {
            d_inputA[insertPos - 1].x = d_data[0].x;
            d_inputA[insertPos - 1].y = d_data[0].y;

        }
        else
        {
            d_inputA[insertPos - 1].x = d_data[posInGrid].x;
            d_inputA[insertPos - 1].y = d_data[posInGrid].y;
        }
    }
}

__global__ void insertZD_kernel(cuDoubleComplex* d_inputA, int inputSize, double* d_data, double* pdblPos, int dataSize, int isScalar)
{
    int posInGrid = blockIdx.x * blockDim.x + threadIdx.x;

    if(posInGrid < dataSize)
    {
        int insertPos = (int)pdblPos[posInGrid];
        if(insertPos <= 0 || insertPos > inputSize)
        {
            d_iErrInsert=-1;
            return;
        }

        if(isScalar)
        {
            d_inputA[insertPos - 1].x = d_data[0];
            d_inputA[insertPos - 1].y = 0;
        }
        else
        {
            d_inputA[insertPos - 1].x = d_data[posInGrid];
            d_inputA[insertPos - 1].y = 0;
        }
    }
}

cudaError_t cudaInsert(double* d_inputA, int inputSize, double* d_data, double* pdblPos, int dataSize, int isScalar, int* piErr)
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
		int dimgrid   = (int)ceil((float)dataSize/threadMax);

        // perform operation
        dim3 block(threadMax, 1, 1);
        dim3 grid(dimgrid, 1, 1);
        initErrInsert<<<1,1>>>();
        insert_kernel<<<grid, block>>>(d_inputA, inputSize, d_data, pdblPos, dataSize, isScalar);

        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        // get error flag from device
        cudaMemcpyFromSymbol(piErr, "d_iErrInsert", sizeof(int), 0, cudaMemcpyDeviceToHost);
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

cudaError_t cudaZInsert(cuDoubleComplex* d_inputA, int inputSize, cuDoubleComplex* d_data, double* pdblPos, int dataSize, int isScalar, int* piErr)
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
		int dimgrid   = (int)ceil((float)dataSize/threadMax);

        // perform operation
        dim3 block(threadMax, 1, 1);
        dim3 grid(dimgrid, 1, 1);
        initErrInsert<<<1,1>>>();
        insertZ_kernel<<<grid, block>>>(d_inputA, inputSize, d_data, pdblPos, dataSize, isScalar);

        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        // get error flag from device
        cudaMemcpyFromSymbol(piErr, "d_iErrInsert", sizeof(int), 0, cudaMemcpyDeviceToHost);
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

cudaError_t cudaZDInsert(cuDoubleComplex* d_inputA, int inputSize, double* d_data, double* pdblPos, int dataSize, int isScalar, int* piErr)
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
		int dimgrid   = (int)ceil((float)dataSize/threadMax);

        // perform operation
        dim3 block(threadMax, 1, 1);
        dim3 grid(dimgrid, 1, 1);
        initErrInsert<<<1,1>>>();
        insertZD_kernel<<<grid, block>>>(d_inputA, inputSize, d_data, pdblPos, dataSize, isScalar);

        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        // get error flag from device
        cudaMemcpyFromSymbol(piErr, "d_iErrInsert", sizeof(int), 0, cudaMemcpyDeviceToHost);
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
