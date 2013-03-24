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
/* ========================================================================== */
#include "gpuPointerManager.hxx"
#include "config_gpu.h"
#include "gpu_wrap.h"
#include "useCuda.h"
#include "gw_gpu.h"
#include "checkDevice.h"
#include "gpuContext.hxx"
/* ========================================================================== */
#include "api_scilab.h"
#include "Scierror.h"
#include "sciprint.h"
/* ========================================================================== */
#ifdef WITH_CUDA
#include "pointerCuda.hxx"
#endif

#ifdef WITH_OPENCL
#include "pointerOpenCL.hxx"
#endif

int sci_gpuClone(char *fname)
{
    CheckLhs(1, 1);
    CheckRhs(1, 1);

    void* pvPtr = NULL;
    int* piAddr = NULL;
    SciErr sciErr;
    int inputType;

    try
    {
        if (!isGpuInit())
        {
            throw "gpu is not initialised. Please launch gpuInit() before use this function.";
        }

        sciErr = getVarAddressFromPosition(pvApiCtx, 1, &piAddr);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        sciErr = getVarType(pvApiCtx, piAddr, &inputType);
        if (inputType != sci_pointer)
        {
            throw "gpuClone : Bad type for input argument #1 : A GPU pointer expected.";
        }

        sciErr = getPointer(pvApiCtx, piAddr, (void**)&pvPtr);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        GpuPointer* gmat = (GpuPointer*)(pvPtr);

        if (!PointerManager::getInstance()->findGpuPointerInManager(gmat))
        {
            throw "gpuClone : Bad type for input argument #1. Only variables created with GPU functions allowed.";
        }

        if (useCuda() && gmat->getGpuType() != GpuPointer::CudaType)
        {
            throw "gpuClone : Bad type for input argument #1: A Cuda pointer expected.";
        }

        if (useCuda() == false && gmat->getGpuType() != GpuPointer::OpenCLType)
        {
            throw "gpuClone : Bad type for input argument #1: A OpenCL pointer expected.";
        }

        GpuPointer* gclone = gmat->clone();

        PointerManager::getInstance()->addGpuPointerInManager(gclone);
        sciErr = createPointer(pvApiCtx, Rhs + 1, (void*)gclone);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        LhsVar(1) = Rhs + 1;
        PutLhsVar();

        return 0;
    }
#ifdef WITH_CUDA
    catch (cudaError_t cudaE)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaE);
    }
#endif
    catch (const char* str)
    {
        Scierror(999, "%s\n", str);
    }
    catch (SciErr E)
    {
        printError(&E, 0);
    }

    return EXIT_FAILURE;
}
/* ========================================================================== */
