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
#include <stdio.h>
#include "matrixTranspose.h"

__global__ void transpose(double *odata, double* idata, int width, int height)// transpose a complex matrix stored in colonne
{
   unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

   if (xIndex < width && yIndex < height)
   {
       unsigned int index_in  = yIndex + height * xIndex;
       unsigned int index_out = xIndex + width * yIndex;
       odata[index_out] = idata[index_in];
   }
}
__global__ void transposeZ(cuDoubleComplex *odata, cuDoubleComplex *idata, int width, int height)// transpose a complex matrix stored in colonne
{
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    if (xIndex < width && yIndex < height)
    {
        unsigned int index_in  = yIndex + height * xIndex;
        unsigned int index_out = xIndex + width * yIndex;
        odata[index_out].x = idata[index_in].x;
        odata[index_out].y = -idata[index_in].y;
    }
}

cudaError_t cudaTranspose(double* d_input, double* d_output, int rows, int cols)
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

        transpose<<<grid, block>>>(d_output, d_input, cols, rows);

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
cudaError_t cudaZTranspose(cuDoubleComplex* d_input, cuDoubleComplex* d_output, int rows, int cols)
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

        transposeZ<<<grid, block>>>(d_output, d_input, cols, rows);

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

