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
#include "cudaRealImgToComplex.hxx"
#include "makecucomplex.h"

PointerCuda* cudaRealImgToComplex(PointerCuda* gpuPtrA, PointerCuda* gpuPtrB)
{
    cudaError_t cudaStat;

    int irows = gpuPtrA->getRows();
    int icols = gpuPtrA->getCols();

    double* pdblA = gpuPtrA->getGpuPtr();
    double* pdblB = gpuPtrB->getGpuPtr();

    PointerCuda* result = new PointerCuda(irows, icols, true);
    double* pdblReult = result->getGpuPtr();

    cudaStat = createcucomplex(pdblA, pdblB, irows, icols, (cuDoubleComplex*)pdblReult);
    if (cudaStat != cudaSuccess)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaStat);
    }

    cudaThreadSynchronize();

    return result;
}
