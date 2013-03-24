/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
* Copyright (C) DIGITEO - 2010-2011 - Cedric DELAMARRE
* Copyright (C) DIGITEO - 2011 - Allan CORNET
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/
/* ========================================================================== */
#include "config_gpu.h"
#include "gpuPointerManager.hxx"
#ifdef WITH_CUDA
#include "pointerCuda.hxx"
#endif
#ifdef WITH_OPENCL
#include "pointerOpenCL.hxx"
#endif
/* ========================================================================== */
#include "gpu_wrap.h"
#include "useCuda.h"
#include "gw_gpu.h"
#include "checkDevice.h"
/* ========================================================================== */
#include "Scierror.h"
#include "sciprint.h"
#include "api_scilab.h"

int sci_gpuFree(char *fname)
{
    CheckRhs(1, 1);
    CheckLhs(0, 1);

    try
    {
        void *dptr = NULL;
        SciErr sciErr;
        int* piAddr = NULL;
        double zero = 0.;
        int inputType_A = 0;

        if (!isGpuInit())
        {
            throw "gpu is not initialised. Please launch gpuInit() before use this function.";
        }

        sciErr = getVarAddressFromPosition(pvApiCtx, 1, &piAddr);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        sciErr = getVarType(pvApiCtx, piAddr, &inputType_A);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        if (inputType_A != sci_pointer)
        {
            throw "gpuFree : Bad type for input argument #1 : A GPU pointer expected.";
        }

        sciErr = getVarAddressFromPosition(pvApiCtx, 1, &piAddr);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        sciErr = getPointer(pvApiCtx, piAddr, (void**)&dptr);

        if (sciErr.iErr)
        {
            throw sciErr;
        }

        GpuPointer* ptrGpu = (GpuPointer*)dptr;
        if (!PointerManager::getInstance()->findGpuPointerInManager(ptrGpu))
        {
            throw "gpuFree : Bad type for input argument #1. Only variables created with GPU functions allowed.";
        }

        if (useCuda() && ptrGpu->getGpuType() != GpuPointer::CudaType)
        {
            throw "gpuFree : Bad type for input argument #1: A Cuda pointer expected.";
        }

        if (useCuda() == false && ptrGpu->getGpuType() != GpuPointer::OpenCLType)
        {
            throw "gpuFree : Bad type for input argument #1: A OpenCL pointer expected.";
        }

        PointerManager::getInstance()->removeGpuPointerInManager(ptrGpu);
        delete ptrGpu;
        ptrGpu = NULL;

        dptr = NULL;

        createScalarDouble(pvApiCtx, Rhs + 1, zero);
        LhsVar(1) = Rhs + 1;
        PutLhsVar();
    }
    catch (const char* str)
    {
        Scierror(999, "%s\n", str);
    }
    catch (SciErr E)
    {
        printError(&E, 0);
    }
    return 0;
}
/* ========================================================================== */
