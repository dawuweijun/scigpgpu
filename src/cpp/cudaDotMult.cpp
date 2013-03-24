/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) 2013 - Scilab Enterprises - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/
#include "cudaDotMult.hxx"
#include "dotmult.h"

PointerCuda* cudaDotMult(PointerCuda* gpuPtrA, PointerCuda* gpuPtrB)
{
    cudaError_t cudaStat;

    int iSize = gpuPtrA->getSize();
    double* pdblA = gpuPtrA->getGpuPtr();
    double* pdblB = gpuPtrB->getGpuPtr();
    bool bComplexA = gpuPtrA->isGpuComplex();
    bool bComplexB = gpuPtrB->isGpuComplex();

    bool bComplex = false;
    if (bComplexA || bComplexB)
    {
        bComplex = true;
    }

    PointerCuda* result = new PointerCuda(gpuPtrA->getRows(), gpuPtrA->getCols(), bComplex);
    double* pdblReult = result->getGpuPtr();

    if (bComplexA == false && bComplexB == false)
    {
        cudaStat = cudaDotMult(iSize, pdblA, pdblB, pdblReult);

        if (cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaStat);
        }
    }
    else if (bComplexA && bComplexB == false)
    {
        cudaStat = cudaZDDotMult(iSize, (cuDoubleComplex*)pdblA, pdblB, (cuDoubleComplex*)pdblReult);

        if (cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaStat);
        }
    }
    else if (bComplexA == false && bComplexB)
    {
        cudaStat = cudaZDDotMult(iSize, (cuDoubleComplex*)pdblB, pdblA, (cuDoubleComplex*)pdblReult);

        if (cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaStat);
        }
    }
    else // A is complex and B is complex
    {
        cudaStat = cudaZDotMult(iSize, (cuDoubleComplex*)pdblA, (cuDoubleComplex*)pdblB, (cuDoubleComplex*)pdblReult);

        if (cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaStat);
        }
    }

    cudaThreadSynchronize();

    return result;
}
