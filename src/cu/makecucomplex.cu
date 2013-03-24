/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/

#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include "makecucomplex.h"

// compile .cu -> .cpp : nvcc -g -arch sm_13 -cuda sci_makecucomplex.cu

//__global__ void writeToCucomplex(double* d,double* di,int rows,int cols,cuDoubleComplex* d_data, size_t pitch)
__global__ void writeToCucomplex(double* d,double* di,int rows,int cols,cuDoubleComplex* d_data)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

//    int posRead = y*(pitch/sizeof(double))+x;
//    int posWrite = y*cols+x;
    int pos = y*cols+x;

    if((x < cols) && (y < rows))
    {
        d_data[pos].x=d[pos];
        if(di != NULL)
        {
            d_data[pos].y=di[pos];
        }
        else
        {
            d_data[pos].y=0;
        }
	}
}
//__global__ void readInCucomplex(double* d,double* di,int rows,int cols,cuDoubleComplex* d_data,size_t pitch)
__global__ void readInCucomplex(double* d,double* di,int rows,int cols,cuDoubleComplex* d_data)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

//    int posWrite = y*(pitch/sizeof(double))+x;
//    int posRead = y*cols+x;
    int pos = y*cols+x;

    if((x < cols) && (y < rows))
    {
//        d[posWrite]=d_data[posRead].x;
//        di[posWrite]=d_data[posRead].y;
        d[pos]=d_data[pos].x;
        di[pos]=d_data[pos].y;
    }
}

cudaError_t rewritecucomplex(double* d,int rows,int cols, cuDoubleComplex* d_data)
{
    double* di	= NULL;

    int dimblockX = 1;
    int dimgridX  = 1;
    int dimblockY = 1;
    int dimgridY  = 1;

//    size_t R_pitch = cols*sizeof(double);

    cudaError_t cudaStat = cudaGetLastError();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaStat = cudaGetLastError();
    if (cudaStat != cudaSuccess) throw cudaStat;

    int BLOCK_DIM = ( (int)(sqrt((float)deviceProp.maxThreadsPerBlock) / 16) ) * 16;

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

//        writeToCucomplex<<<grid, block>>>(d,di,rows,cols,d_data,R_pitch);
        writeToCucomplex<<<grid, block>>>(d,di,rows,cols,d_data);

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

cudaError_t createcucomplex(double* d,double* di,int rows,int cols, cuDoubleComplex* d_data)
{
    int dimblockX = 1;
    int dimgridX  = 1;
    int dimblockY = 1;
    int dimgridY  = 1;

//    size_t R_pitch = cols*sizeof(double);

    cudaError_t cudaStat = cudaGetLastError();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaStat = cudaGetLastError();
    if (cudaStat != cudaSuccess) throw cudaStat;

    int BLOCK_DIM = ( (int)(sqrt((float)deviceProp.maxThreadsPerBlock) / 16) ) * 16;

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

//        writeToCucomplex<<<grid, block>>>(d,di,rows,cols,d_data,R_pitch);
        writeToCucomplex<<<grid, block>>>(d,di,rows,cols,d_data);

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

// h and hi are host pointer
cudaError_t writecucomplex(double* h,double* hi,int rows,int cols, cuDoubleComplex* d_data)
{
    double* d	= NULL;
    double* di	= NULL;

    int dimblockX = 1;
    int dimgridX  = 1;
    int dimblockY = 1;
    int dimgridY  = 1;

//    size_t R_pitch,I_pitch;

    cudaError_t cudaStat = cudaGetLastError();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaStat = cudaGetLastError();
    if (cudaStat != cudaSuccess) throw cudaStat;

    int BLOCK_DIM = ( (int)(sqrt((float)deviceProp.maxThreadsPerBlock) / 16) ) * 16;

    try
    {
//        cudaMallocPitch((void**)&d, &R_pitch, sizeof(double)*cols,rows);
        cudaMalloc((void**)&d, sizeof(double)*cols*rows);
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

//        cudaMemcpy2D(d, R_pitch,h, cols*sizeof(double), cols*sizeof(double), rows, cudaMemcpyHostToDevice);
        cudaMemcpy(d, h, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        if(hi != NULL)
        {
//            cudaMallocPitch((void**)&di, &I_pitch, sizeof(double)*cols,rows);
            cudaMalloc((void**)&di, sizeof(double)*cols*rows);
            cudaStat = cudaGetLastError();
            if (cudaStat != cudaSuccess) throw cudaStat;
//            cudaMemcpy2D(di, I_pitch,hi, cols*sizeof(double), cols*sizeof(double), rows, cudaMemcpyHostToDevice);
            cudaMemcpy(di, hi, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
            cudaStat = cudaGetLastError();
            if (cudaStat != cudaSuccess) throw cudaStat;
        }

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

//        writeToCucomplex<<<grid, block>>>(d,di,rows,cols,d_data,R_pitch);
        writeToCucomplex<<<grid, block>>>(d,di,rows,cols,d_data);

        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        cudaStat = cudaThreadSynchronize();
        if (cudaStat != cudaSuccess) throw cudaStat;

        cudaFree(d);
        cudaFree(di);
        return cudaSuccess;
    }
    catch(cudaError_t cudaE)
    {
        if(d != NULL) cudaFree(d);
        if(di != NULL) cudaFree(di);
        return cudaE;
    }
}

cudaError_t readcucomplex(double* h,double* hi,int rows,int cols, cuDoubleComplex* d_data)
{
    double* d  = NULL;
    double* di = NULL;

    int dimblockX = 1;
    int dimgridX  = 1;
    int dimblockY = 1;
    int dimgridY  = 1;

//    size_t R_pitch,I_pitch;

    cudaError_t cudaStat = cudaGetLastError();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaStat = cudaGetLastError();
    if (cudaStat != cudaSuccess) throw cudaStat;

    int BLOCK_DIM = ( (int)(sqrt((float)deviceProp.maxThreadsPerBlock) / 16) ) * 16;

    try
    {
//        cudaMallocPitch((void**)&d, &R_pitch, sizeof(double)*cols,rows);
        cudaMalloc((void**)&d, sizeof(double)*cols*rows);
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

//        cudaMallocPitch((void**)&di, &I_pitch, sizeof(double)*cols,rows);
        cudaMalloc((void**)&di, sizeof(double)*cols*rows);
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

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

//        readInCucomplex<<<grid, block>>>(d,di,rows,cols,d_data,R_pitch);
        readInCucomplex<<<grid, block>>>(d,di,rows,cols,d_data);

        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        cudaStat = cudaThreadSynchronize();
        if (cudaStat != cudaSuccess) throw cudaStat;

//        cudaMemcpy2D(h,cols*sizeof(double),d,R_pitch,cols*sizeof(double),rows,cudaMemcpyDeviceToHost);
        cudaMemcpy(h, d, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;
//        cudaMemcpy2D(hi,cols*sizeof(double),di,I_pitch,cols*sizeof(double),rows,cudaMemcpyDeviceToHost);
        cudaMemcpy(hi, di, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        cudaFree(d);
        cudaFree(di);
        return cudaSuccess;
    }
    catch(cudaError_t cudaE)
    {
        if(d != NULL) cudaFree(d);
        if(di != NULL) cudaFree(di);
        return cudaE;
    }
}
