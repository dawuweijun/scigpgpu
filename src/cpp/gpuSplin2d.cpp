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
#include "splin2d.h"
#include "strictIncreasing.h"
#endif
#ifdef WITH_OPENCL
#include "pointerOpenCL.hxx"
#endif
#include "gpuSplin2d.hxx"
#include "useCuda.h"

GpuPointer* gpuSplin2d(GpuPointer* gpuPtrX, GpuPointer* gpuPtrY, GpuPointer* gpuPtrZ, SplineType spType)
{
    int sizeOfX = gpuPtrX->getSize();
    int sizeOfY = gpuPtrY->getSize();
    int sizeOfC = 0;

#ifdef WITH_CUDA
    if (useCuda())
    {
        cudaError_t cudaStat = cudaSuccess;

        // verify strict increasing order for x and y
        int isStrictIncreasing = 0;
        cudaStat = cudaStrictIncreasing(gpuPtrX->getGpuPtr(), sizeOfX, &isStrictIncreasing);
        if (cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaStat);
        }
        if (isStrictIncreasing == 0)
        {
            return NULL;
        }

        cudaStat = cudaStrictIncreasing(gpuPtrY->getGpuPtr(), sizeOfY, &isStrictIncreasing);
        if (cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaStat);
        }
        if (isStrictIncreasing == 0)
        {
            return NULL;
        }

        sizeOfC = 16 * (sizeOfX - 1) * (sizeOfY - 1);
        PointerCuda* gpuPtrC = new PointerCuda(sizeOfC, 1, false);

        // work space
        PointerCuda* gpuPtrP = new PointerCuda(sizeOfX, sizeOfY, false);
        PointerCuda* gpuPtrQ = new PointerCuda(sizeOfX, sizeOfY, false);
        PointerCuda* gpuPtrR = new PointerCuda(sizeOfX, sizeOfY, false);

        if (spType == MONOTONE || spType == FAST || spType == FAST_PERIODIC)
        {
            cudaStat = cudaBicubicSubSplin(((PointerCuda*)gpuPtrX)->getGpuPtr(),
                                           ((PointerCuda*)gpuPtrY)->getGpuPtr(),
                                           ((PointerCuda*)gpuPtrZ)->getGpuPtr(),
                                           sizeOfX, sizeOfY,
                                           gpuPtrP->getGpuPtr(), gpuPtrQ->getGpuPtr(),
                                           gpuPtrR->getGpuPtr(), spType, gpuPtrC->getGpuPtr());
        }
        else // not_a_knot, natural, periodic
        {
            // work space
            PointerCuda* gpuPtrAd    = new PointerCuda(sizeOfX, sizeOfY, false);
            PointerCuda* gpuPtrAsd   = new PointerCuda(sizeOfX, sizeOfY, false);
            PointerCuda* gpuPtrQdu   = new PointerCuda(sizeOfX, sizeOfY, false);

            cudaStat = cudaBicubicSplin(((PointerCuda*)gpuPtrX)->getGpuPtr(),
                                        ((PointerCuda*)gpuPtrY)->getGpuPtr(),
                                        ((PointerCuda*)gpuPtrZ)->getGpuPtr(),
                                        sizeOfX, sizeOfY,
                                        gpuPtrP->getGpuPtr(),   gpuPtrQ->getGpuPtr(),
                                        gpuPtrR->getGpuPtr(),   gpuPtrAd->getGpuPtr(),
                                        gpuPtrAsd->getGpuPtr(), gpuPtrQdu->getGpuPtr(),
                                        spType, gpuPtrC->getGpuPtr());

            delete gpuPtrAd;
            delete gpuPtrAsd;
            delete gpuPtrQdu;
        }

        delete gpuPtrP;
        delete gpuPtrQ;
        delete gpuPtrR;

        if (cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaStat);
        }

        cudaThreadSynchronize();
        return gpuPtrC;
    }
#endif

#ifdef WITH_OPENCL
    if (!useCuda())
    {
        return NULL;
    }
#endif
}
