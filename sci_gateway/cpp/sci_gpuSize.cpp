/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
* Copyright (C) DIGITEO - 2010-2012 - Cedric DELAMARRE
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
#include "gpuPointerManager.hxx"
#ifdef WITH_CUDA
#include "pointerCuda.hxx"
#endif

#ifdef WITH_OPENCL
#include "pointerOpenCL.hxx"
#endif
/* ========================================================================== */

int sci_gpuSize(char *fname)
{
    CheckLhs(1, 2);
    CheckRhs(1, 1);

    int* piAddr = NULL;
    void* dptr = NULL;
    int inputType;
    double r = 0;
    double c = 0;
    SciErr sciErr;

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
        if (sciErr.iErr)
        {
            throw sciErr;
        }
        if (inputType != sci_pointer)
        {
            throw "gpuSize : Bad type for input argument #1: A GPU matrix expected.";
        }

        sciErr = getPointer(pvApiCtx, piAddr, (void**)&dptr);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        GpuPointer* gmat = (GpuPointer*)(dptr);

        if (!PointerManager::getInstance()->findGpuPointerInManager(gmat))
        {
            throw "gpuSize : Bad type for input argument #1: Variables created with GPU functions expected.";
        }

        if (useCuda() && gmat->getGpuType() != GpuPointer::CudaType)
        {
            throw "gpuSize : Bad type for input argument #1: A Cuda pointer expected.";
        }

        if (useCuda() == false && gmat->getGpuType() != GpuPointer::OpenCLType)
        {
            throw "gpuSize : Bad type for input argument #1: A OpenCL pointer expected.";
        }

        r = gmat->getRows();
        c = gmat->getCols();

        if (nbOutputArgument(pvApiCtx) == 2)
        {
            sciErr = createMatrixOfDouble(pvApiCtx, Rhs + 1, 1, 1, &r);
            if (sciErr.iErr)
            {
                throw sciErr;
            }
            sciErr = createMatrixOfDouble(pvApiCtx, Rhs + 2, 1, 1, &c);
            if (sciErr.iErr)
            {
                throw sciErr;
            }
            LhsVar(1) = Rhs + 1;
            LhsVar(2) = Rhs + 2;
        }
        else
        {
            double sizes[2] = {r, c};
            sciErr = createMatrixOfDouble(pvApiCtx, Rhs + 1, 1, 2, sizes);
            if (sciErr.iErr)
            {
                throw sciErr;
            }
            LhsVar(1) = Rhs + 1;
        }

        PutLhsVar();
        return 0;
    }
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
