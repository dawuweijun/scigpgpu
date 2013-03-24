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
#include "cudaElemMin.hxx"
#include "elemWiseMin.h"

void cudaElemMin(PointerCuda* gpuPtrA, PointerCuda* gpuPtrB, PointerCuda* gpuPtrRes)
{
    cudaError_t cudaStat;
    if (gpuPtrA->isGpuComplex() == false && gpuPtrB->isGpuComplex() == false)
    {
        cudaStat = cudaMinElementwise(gpuPtrA->getGpuPtr(),
                                      gpuPtrB->getGpuPtr(),
                                      gpuPtrRes->getGpuPtr(),
                                      gpuPtrA->getRows(),
                                      gpuPtrA->getCols());

        if (cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaStat);
        }

    }
    else if (gpuPtrA->isGpuComplex() && gpuPtrB->isGpuComplex() == false)
    {
        cudaStat = cudaZDMinElementwise((cuDoubleComplex*)gpuPtrA->getGpuPtr(),
                                        gpuPtrB->getGpuPtr(),
                                        (cuDoubleComplex*)gpuPtrRes->getGpuPtr(),
                                        gpuPtrA->getRows(),
                                        gpuPtrA->getCols());

        if (cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaStat);
        }
    }
    else if (gpuPtrA->isGpuComplex() == false && gpuPtrB->isGpuComplex())
    {
        cudaStat = cudaZDMinElementwise((cuDoubleComplex*)gpuPtrB->getGpuPtr(),
                                        gpuPtrA->getGpuPtr(),
                                        (cuDoubleComplex*)gpuPtrRes->getGpuPtr(),
                                        gpuPtrA->getRows(),
                                        gpuPtrA->getCols());

        if (cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaStat);
        }
    }
    else // A is complex and B is complex
    {
        cudaStat = cudaZMinElementwise( (cuDoubleComplex*)gpuPtrA->getGpuPtr(),
                                        (cuDoubleComplex*)gpuPtrB->getGpuPtr(),
                                        (cuDoubleComplex*)gpuPtrRes->getGpuPtr(),
                                        gpuPtrA->getRows(),
                                        gpuPtrA->getCols());

        if (cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaStat);
        }
    }

    cudaThreadSynchronize();
}
