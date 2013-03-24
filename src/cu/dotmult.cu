/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2011 - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/

#include <math.h>
#include "dotmult.h"

__global__ void dotmult_kernel(double* dA, double* dB, int elems, double* result)
{
	int posInGrid  = blockIdx.x  * blockDim.x + threadIdx.x;

	if(posInGrid < elems)
	{
        result[posInGrid] = dA[posInGrid] * dB[posInGrid];
	}
}

__global__ void dotmultZ_kernel(cuDoubleComplex* dA, cuDoubleComplex* dB, int elems, cuDoubleComplex* result)
{
	int posInGrid  = blockIdx.x  * blockDim.x + threadIdx.x;

	if(posInGrid < elems)
	{
        result[posInGrid].x = dA[posInGrid].x * dB[posInGrid].x - dA[posInGrid].y * dB[posInGrid].y;
        result[posInGrid].y = dA[posInGrid].x * dB[posInGrid].y + dA[posInGrid].y * dB[posInGrid].x;
	}
}

__global__ void dotmultZD_kernel(cuDoubleComplex* dA, double* dB, int elems, cuDoubleComplex* result)
{
	int posInGrid  = blockIdx.x  * blockDim.x + threadIdx.x;

	if(posInGrid < elems)
	{
        result[posInGrid].x = dA[posInGrid].x * dB[posInGrid];
        result[posInGrid].y = dA[posInGrid].y * dB[posInGrid];
	}
}

cudaError_t cudaDotMult(int elems, double* dA, double* dB, double* dRes)
{
	cudaError_t cudaStat = cudaGetLastError();
	try
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess) throw cudaStat;

		int threadMax 	= deviceProp.maxThreadsDim[0];
		int dimgrid  	= (int) ceil((float)elems/threadMax);

        dim3 block(threadMax, 1, 1);
        dim3 grid(dimgrid, 1, 1);
        dotmult_kernel<<<grid, block>>>(dA, dB, elems, dRes);

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

cudaError_t cudaZDotMult(int elems, cuDoubleComplex* dA, cuDoubleComplex* dB, cuDoubleComplex* dRes)
{
	cudaError_t cudaStat = cudaGetLastError();
	try
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess) throw cudaStat;

		int threadMax 	= deviceProp.maxThreadsDim[0];
		int dimgrid  	= (int) ceil((float)elems/threadMax);

        dim3 block(threadMax, 1, 1);
        dim3 grid(dimgrid, 1, 1);
        dotmultZ_kernel<<<grid, block>>>(dA, dB, elems, dRes);

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

cudaError_t cudaZDDotMult(int elems, cuDoubleComplex* dA, double* dB, cuDoubleComplex* dRes)
{
	cudaError_t cudaStat = cudaGetLastError();
	try
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess) throw cudaStat;

		int threadMax 	= deviceProp.maxThreadsDim[0];
		int dimgrid  	= (int) ceil((float)elems/threadMax);

        dim3 block(threadMax, 1, 1);
        dim3 grid(dimgrid, 1, 1);
        dotmultZD_kernel<<<grid, block>>>(dA, dB, elems, dRes);

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
