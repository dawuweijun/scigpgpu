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

/*****************/
/*    A .*. B    */
/*****************/
#include <math.h>
#include <cuComplex.h>
#include "kronecker.h"

// number of threads = size of B, kernel loop on A
__global__ void kroneckerB_kernel(double* dA, int iRowsA, int iColsA, bool bAIsComplex, double* dB, int iRowsB, int iColsB, bool bBIsComplex, double* dOut, int iRowsOut)
{
    int iPos = blockIdx.x * blockDim.x + threadIdx.x;

    if(iPos >= iRowsB*iColsB)
    {
        return;
    }

    int iPosX = iPos / iRowsB;
    int iPosY = iPos % iRowsB;

    if(bAIsComplex == false && bBIsComplex == false)
    {// real case
        for(int i = 0; i < iColsA; i++)
        {
            for(int j = 0; j < iRowsA; j++)
            {
                int pos = (i * iColsB + iPosX) * iRowsOut + j * iRowsB + iPosY;
                dOut[pos] = dA[i*iRowsA+j] * dB[iPos];
            }
        }
    }
    else if(bAIsComplex && bBIsComplex == false)
    {// A complex
        cuDoubleComplex* out = (cuDoubleComplex*)dOut;
        cuDoubleComplex* da  = (cuDoubleComplex*)dA;
        for(int i = 0; i < iColsA; i++)
        {
            for(int j = 0; j < iRowsA; j++)
            {
                int iPosOut = (i * iColsB + iPosX) * iRowsOut + j * iRowsB + iPosY;
                int iPosA = i*iRowsA+j;
                out[iPosOut].x = da[iPosA].x * dB[iPos];
                out[iPosOut].y = da[iPosA].y * dB[iPos];
            }
        }
    }
    else if(bAIsComplex == false && bBIsComplex)
    {// B complex
        cuDoubleComplex* out = (cuDoubleComplex*)dOut;
        cuDoubleComplex* db  = (cuDoubleComplex*)dB;
        for(int i = 0; i < iColsA; i++)
        {
            for(int j = 0; j < iRowsA; j++)
            {
                int iPosOut = (i * iColsB + iPosX) * iRowsOut + j * iRowsB + iPosY;
                int iPosA = i*iRowsA+j;
                out[iPosOut].x = dA[iPosA] * db[iPos].x;
                out[iPosOut].y = dA[iPosA] * db[iPos].y;
            }
        }
    }
    else
    {// A and B complex
        cuDoubleComplex* out = (cuDoubleComplex*)dOut;
        cuDoubleComplex* da  = (cuDoubleComplex*)dA;
        cuDoubleComplex* db  = (cuDoubleComplex*)dB;
        for(int i = 0; i < iColsA; i++)
        {
            for(int j = 0; j < iRowsA; j++)
            {
                int iPosOut = (i * iColsB + iPosX) * iRowsOut + j * iRowsB + iPosY;
                int iPosA = i*iRowsA+j;
                out[iPosOut].x = da[iPosA].x * db[iPos].x - da[iPosA].y * db[iPos].y;
                out[iPosOut].y = da[iPosA].x * db[iPos].y + da[iPosA].y * db[iPos].x;
            }
        }
    }
}

// number of threads = size of A, kernel loop on B
__global__ void kroneckerA_kernel(double* dA, int iRowsA, int iColsA, bool bAIsComplex, double* dB, int iRowsB, int iColsB, bool bBIsComplex, double* dOut, int iRowsOut)
{
    int iPos = blockIdx.x * blockDim.x + threadIdx.x;

    if(iPos >= iRowsA*iColsA)
    {
        return;
    }

    int iPosX = iPos / iRowsA;
    int iPosY = iPos % iRowsA;

    if(bAIsComplex == false && bBIsComplex == false)
    {// real case
        for(int i = 0; i < iColsB; i++)
        {
            for(int j = 0; j < iRowsB; j++)
            {
                int iPosOut = (iPosX * iColsB + i) * iRowsOut + iPosY * iRowsB + j;
                dOut[iPosOut] = dA[iPos] * dB[i*iRowsB+j];
            }
        }
    }
    else if(bAIsComplex && bBIsComplex == false)
    {// A complex
        cuDoubleComplex* out = (cuDoubleComplex*)dOut;
        cuDoubleComplex* da  = (cuDoubleComplex*)dA;
        for(int i = 0; i < iColsB; i++)
        {
            for(int j = 0; j < iRowsB; j++)
            {
                int iPosOut = (iPosX * iColsB + i) * iRowsOut + iPosY * iRowsB + j;
                int iPosB = i*iRowsB+j;
                out[iPosOut].x = da[iPos].x * dB[iPosB];
                out[iPosOut].y = da[iPos].y * dB[iPosB];
            }
        }
    }
    else if(bAIsComplex == false && bBIsComplex)
    {// B complex
        cuDoubleComplex* out = (cuDoubleComplex*)dOut;
        cuDoubleComplex* db  = (cuDoubleComplex*)dB;
        for(int i = 0; i < iColsB; i++)
        {
            for(int j = 0; j < iRowsB; j++)
            {
                int iPosOut = (iPosX * iColsB + i) * iRowsOut + iPosY * iRowsB + j;
                int iPosB = i*iRowsB+j;
                out[iPosOut].x = dA[iPos] * db[iPosB].x;
                out[iPosOut].y = dA[iPos] * db[iPosB].y;
            }
        }
    }
    else
    {// A and B complex
        cuDoubleComplex* out = (cuDoubleComplex*)dOut;
        cuDoubleComplex* da  = (cuDoubleComplex*)dA;
        cuDoubleComplex* db  = (cuDoubleComplex*)dB;
        for(int i = 0; i < iColsB; i++)
        {
            for(int j = 0; j < iRowsB; j++)
            {
                int iPosOut = (iPosX * iColsB + j) * iRowsOut + iPosY * iRowsB + i;
                int iPosB = i*iRowsB+j;
                out[iPosOut].x = da[iPos].x * db[iPosB].x - da[iPos].y * db[iPosB].y;
                out[iPosOut].y = da[iPos].x * db[iPosB].y + da[iPos].y * db[iPosB].x;
            }
        }
    }
}

// grid size = size of A, Block size = size of B
__global__ void kroneckerAB_kernel(double* dA, bool bAIsComplex, double* dB, bool bBIsComplex, double* dOut, int iRowsOut)
{
    int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
    int iPosY = blockIdx.y * blockDim.y + threadIdx.y;

    int iPosA = blockIdx.x * gridDim.y + blockIdx.y;
    int iPosB = threadIdx.x * blockDim.y + threadIdx.y;

    if(bAIsComplex == false && bBIsComplex == false)
    {// real case
        dOut[iPosX * iRowsOut + iPosY] = dA[iPosA] * dB[iPosB];
    }
    else if(bAIsComplex && bBIsComplex == false)
    {// A complex
        cuDoubleComplex* out = (cuDoubleComplex*)dOut;
        cuDoubleComplex* da  = (cuDoubleComplex*)dA;
        out[iPosX * iRowsOut + iPosY].x = da[iPosA].x * dB[iPosB];
        out[iPosX * iRowsOut + iPosY].y = da[iPosA].y * dB[iPosB];
    }
    else if(bAIsComplex == false && bBIsComplex)
    {// B complex
        cuDoubleComplex* out = (cuDoubleComplex*)dOut;
        cuDoubleComplex* db  = (cuDoubleComplex*)dB;
        out[iPosX * iRowsOut + iPosY].x = dA[iPosA] * db[iPosB].x;
        out[iPosX * iRowsOut + iPosY].y = dA[iPosA] * db[iPosB].y;
    }
    else
    {// A and B complex
        cuDoubleComplex* out = (cuDoubleComplex*)dOut;
        cuDoubleComplex* da  = (cuDoubleComplex*)dA;
        cuDoubleComplex* db  = (cuDoubleComplex*)dB;
        out[iPosX * iRowsOut + iPosY].x = da[iPosA].x * db[iPosB].x - da[iPosA].y * db[iPosB].y;
        out[iPosX * iRowsOut + iPosY].y = da[iPosA].x * db[iPosB].y + da[iPosA].y * db[iPosB].x;
    }
}

cudaError_t cudaKronecker(double* dA, int iRowsA, int iColsA, bool bAIsComplex, double* dB, int iRowsB, int iColsB, bool bBIsComplex, double* dOut)
{
    cudaError_t cudaStat = cudaGetLastError();
    int iSizeA = iRowsA * iColsA;
    int iSizeB = iRowsB * iColsB;

    try
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        int threadMax = deviceProp.maxThreadsDim[0];
        int blockMax  = deviceProp.maxGridSize[0];

        if(iSizeA > blockMax || iSizeB > threadMax)
        {
            if(iSizeA > iSizeB)
            {
                int iDimBlock = (iSizeA > threadMax) ? threadMax : iSizeA;
                int iDimGrid  = (int)ceil((float)iSizeA/iDimBlock);

                dim3 block(iDimBlock, 1, 1);
                dim3 grid(iDimGrid, 1, 1);
                kroneckerA_kernel<<<grid, block>>>( dA, iRowsA, iColsA, bAIsComplex,
                                                    dB, iRowsB, iColsB, bBIsComplex,
                                                    dOut, iRowsA*iRowsB);
            }
            else
            {
                int iDimBlock = (iSizeB > threadMax) ? threadMax : iSizeB;
                int iDimGrid  = (int)ceil((float)iSizeB/iDimBlock);

                dim3 block(iDimBlock, 1, 1);
                dim3 grid(iDimGrid, 1, 1);
                kroneckerB_kernel<<<grid, block>>>( dA, iRowsA, iColsA, bAIsComplex,
                                                    dB, iRowsB, iColsB, bBIsComplex,
                                                    dOut, iRowsA*iRowsB);
            }
        }
        else
        {
            dim3 block(iColsB, iRowsB, 1);
            dim3 grid(iColsA, iRowsA, 1);
            kroneckerAB_kernel<<<grid, block>>>(dA, bAIsComplex, dB, bBIsComplex, dOut, iRowsA*iRowsB);
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
