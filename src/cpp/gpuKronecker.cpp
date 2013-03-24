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
#include <math.h>
#include "config_gpu.h"
#ifdef WITH_CUDA
#include "pointerCuda.hxx"
#include "kronecker.h"
#endif
#ifdef WITH_OPENCL
#include "pointerOpenCL.hxx"
#endif
#include "gpuKronecker.hxx"
#include "useCuda.h"

GpuPointer* gpuKronecker(GpuPointer* gpuPtrA, GpuPointer* gpuPtrB)
{
    int iRowsA = gpuPtrA->getRows();
    int iColsA = gpuPtrA->getCols();
    int iRowsB = gpuPtrB->getRows();
    int iColsB = gpuPtrB->getCols();
    bool bAIsComplex = gpuPtrA->isGpuComplex();
    bool bBIsComplex = gpuPtrB->isGpuComplex();

#ifdef WITH_CUDA
    if (useCuda())
    {
        cudaError_t cudaStat = cudaSuccess;
        PointerCuda* gpuPtrOut = new PointerCuda(gpuPtrA->getRows()*gpuPtrB->getRows(), gpuPtrA->getCols()*gpuPtrB->getCols(), bAIsComplex || bBIsComplex);

        cudaStat = cudaKronecker(((PointerCuda*)gpuPtrA)->getGpuPtr(), iRowsA, iColsA, bAIsComplex,
                                 ((PointerCuda*)gpuPtrB)->getGpuPtr(), iRowsB, iColsB, bBIsComplex,
                                 gpuPtrOut->getGpuPtr());

        if (cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaStat);
        }

        cudaThreadSynchronize();
        return gpuPtrOut;
    }
#endif

#ifdef WITH_OPENCL
    if (!useCuda())
    {
        return NULL;
    }
#endif
}
