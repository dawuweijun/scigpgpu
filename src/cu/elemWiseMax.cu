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
#include "elemWiseMax.h"

__global__ void elementWiseMax(double *odata, double* idataA, double* idataB, int width, int height)
{
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int index  = yIndex + height * xIndex;

    if (xIndex < width && yIndex < height)
    {
        if(idataA[index] < idataB[index])
        {
            odata[index] = idataB[index];
        }
        else
        {
            odata[index] = idataA[index];
        }
    }
}
__global__ void elementWiseZMax(cuDoubleComplex *odata, cuDoubleComplex* idataA, cuDoubleComplex* idataB, int width, int height)
{
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int index  = yIndex + height * xIndex;
    double valA = 0;
    double valB = 0;

    if (xIndex < width && yIndex < height)
    {
        valA = abs(idataA[index].x) + abs(idataA[index].y);
        valB = abs(idataB[index].x) + abs(idataB[index].y);
        if(valA < valB)
        {
            odata[index].x = idataB[index].x;
            odata[index].y = idataB[index].y;
        }
        else
        {
            odata[index].x = idataA[index].x;
            odata[index].y = idataA[index].y;
        }
    }
}

__global__ void elementWiseZDMax(cuDoubleComplex *odata, cuDoubleComplex* idataA, double* idataB, int width, int height)
{
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int index  = yIndex + height * xIndex;
    double valA = 0;
    double valB = 0;

    if (xIndex < width && yIndex < height)
    {
        valA = abs(idataA[index].x) + abs(idataA[index].y);
        valB = abs(idataB[index]);
        if(valA < valB)
        {
            odata[index].x = idataB[index];
            odata[index].y = 0;
        }
        else
        {
            odata[index].x = idataA[index].x;
            odata[index].y = idataA[index].y;
        }
    }
}

cudaError_t cudaMaxElementwise(double* d_inputA, double* d_inputB, double* d_output, int rows, int cols)
{
    int dimblockX 	= 1;
    int dimgridX  	= 1;
    int dimblockY 	= 1;
    int dimgridY  	= 1;
    int BLOCK_DIM   = 0;

    cudaError_t cudaStat = cudaGetLastError();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaStat = cudaGetLastError();
    if (cudaStat != cudaSuccess) throw cudaStat;

    BLOCK_DIM = ((int)(sqrt((float)deviceProp.maxThreadsPerBlock)/16))*16;

    try
    {
        if(rows*cols > deviceProp.maxThreadsPerBlock)
		{
            if(rows > BLOCK_DIM && cols > BLOCK_DIM)
            {
                dimblockY = BLOCK_DIM;
                dimgridY  = (int) ceil((float)rows/dimblockY);
                dimblockX = (int)((float)deviceProp.maxThreadsPerBlock/dimblockY);
                dimgridX  = (int) ceil((float)cols/dimblockX);
            }
            else if(cols > BLOCK_DIM)
            {
                dimblockY = rows;
                dimblockX = (int)((float)deviceProp.maxThreadsPerBlock/rows);
                dimgridX  = (int)ceil((float)cols/dimblockX);
            }
            else if(rows > BLOCK_DIM)
            {
                dimblockX = cols;
                dimblockY = (int)((float)deviceProp.maxThreadsPerBlock/cols);
                dimgridY  = (int)ceil((float)rows/dimblockY);
            }
		}
		else
		{
			dimblockX = cols;
			dimblockY = rows;
		}

        dim3 block(dimblockX, dimblockY, 1);
        dim3 grid(dimgridX, dimgridY, 1);

        elementWiseMax<<<grid, block>>>(d_output, d_inputA, d_inputB, cols, rows);

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

cudaError_t cudaZMaxElementwise(cuDoubleComplex* d_inputA, cuDoubleComplex* d_inputB, cuDoubleComplex* d_output, int rows, int cols)
{
    int dimblockX 	= 1;
    int dimgridX  	= 1;
    int dimblockY 	= 1;
    int dimgridY  	= 1;
    int BLOCK_DIM   = 0;

    cudaError_t cudaStat = cudaGetLastError();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaStat = cudaGetLastError();
    if (cudaStat != cudaSuccess) throw cudaStat;

    BLOCK_DIM = ((int)(sqrt((float)deviceProp.maxThreadsPerBlock)/16))*16;

    try
    {
        if(rows*cols > deviceProp.maxThreadsPerBlock)
		{
            if(rows > BLOCK_DIM && cols > BLOCK_DIM)
            {
                dimblockY = BLOCK_DIM;
                dimgridY  = (int) ceil((float)rows/dimblockY);
                dimblockX = (int)((float)deviceProp.maxThreadsPerBlock/dimblockY);
                dimgridX  = (int) ceil((float)cols/dimblockX);
            }
            else if(cols > BLOCK_DIM)
            {
                dimblockY = rows;
                dimblockX = (int)((float)deviceProp.maxThreadsPerBlock/rows);
                dimgridX  = (int)ceil((float)cols/dimblockX);
            }
            else if(rows > BLOCK_DIM)
            {
                dimblockX = cols;
                dimblockY = (int)((float)deviceProp.maxThreadsPerBlock/cols);
                dimgridY  = (int)ceil((float)rows/dimblockY);
            }
		}
		else
		{
			dimblockX = cols;
			dimblockY = rows;
		}

        dim3 block(dimblockX, dimblockY, 1);
        dim3 grid(dimgridX, dimgridY, 1);

        elementWiseZMax<<<grid, block>>>(d_output, d_inputA, d_inputB, cols, rows);

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

cudaError_t cudaZDMaxElementwise(cuDoubleComplex* d_inputA, double* d_inputB, cuDoubleComplex* d_output, int rows, int cols)
{
    int dimblockX 	= 1;
    int dimgridX  	= 1;
    int dimblockY 	= 1;
    int dimgridY  	= 1;
    int BLOCK_DIM   = 0;

    cudaError_t cudaStat = cudaGetLastError();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaStat = cudaGetLastError();
    if (cudaStat != cudaSuccess) throw cudaStat;

    BLOCK_DIM = ((int)(sqrt((float)deviceProp.maxThreadsPerBlock)/16))*16;

    try
    {
        if(rows*cols > deviceProp.maxThreadsPerBlock)
		{
            if(rows > BLOCK_DIM && cols > BLOCK_DIM)
            {
                dimblockY = BLOCK_DIM;
                dimgridY  = (int) ceil((float)rows/dimblockY);
                dimblockX = (int)((float)deviceProp.maxThreadsPerBlock/dimblockY);
                dimgridX  = (int) ceil((float)cols/dimblockX);
            }
            else if(cols > BLOCK_DIM)
            {
                dimblockY = rows;
                dimblockX = (int)((float)deviceProp.maxThreadsPerBlock/rows);
                dimgridX  = (int)ceil((float)cols/dimblockX);
            }
            else if(rows > BLOCK_DIM)
            {
                dimblockX = cols;
                dimblockY = (int)((float)deviceProp.maxThreadsPerBlock/cols);
                dimgridY  = (int)ceil((float)rows/dimblockY);
            }
		}
		else
		{
			dimblockX = cols;
			dimblockY = rows;
		}

        dim3 block(dimblockX, dimblockY, 1);
        dim3 grid(dimgridX, dimgridY, 1);

        elementWiseZDMax<<<grid, block>>>(d_output, d_inputA, d_inputB, cols, rows);

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
