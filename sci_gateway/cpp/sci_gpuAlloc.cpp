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
#include "gpuContext.hxx"
/* ========================================================================== */
#include "api_scilab.h"
#include "Scierror.h"
#include "sciprint.h"
/* ========================================================================== */
int sci_gpuAlloc(char *fname)
{
    SciErr sciErr;
    double* dtmp        = NULL;
    int* piAddr_A       = NULL;
    int* piAddr_B       = NULL;
    int trashr  = 0, trashc     = 0;
    int rows    = 0, columns    = 0;

    int inputType_A = 0;
    int inputType_B = 0;

    CheckRhs(2, 2);
    CheckLhs(0, 1);


    try
    {
        if (!isGpuInit())
        {
            throw "gpu is not initialised. Please launch gpuInit() before use this function.";
        }
        sciErr = getVarAddressFromPosition(pvApiCtx, 1, &piAddr_A);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        sciErr = getVarAddressFromPosition(pvApiCtx, 2, &piAddr_B);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        sciErr = getVarType(pvApiCtx, piAddr_A, &inputType_A);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        sciErr = getVarType(pvApiCtx, piAddr_B, &inputType_B);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        if (inputType_A != sci_matrix || isVarComplex(pvApiCtx, piAddr_A))
        {
            throw "gpuAlloc : Bad type for input argument #1: A real scalar expected.";
        }

        if (inputType_B != sci_matrix || isVarComplex(pvApiCtx, piAddr_B))
        {
            throw "gpuAlloc : Bad type for input argument #2: A real scalar expected.";
        }

        sciErr = getMatrixOfDouble(pvApiCtx, piAddr_A, &trashr, &trashc, &dtmp);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        if (trashr * trashc != 1)
        {
            throw "gpuAlloc : Bad size for input argument #1: A scalar expected.";
        }

        rows = static_cast<int>(dtmp[0]);

        sciErr = getMatrixOfDouble(pvApiCtx, piAddr_B, &trashr, &trashc, &dtmp);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        if (trashr * trashc != 1)
        {
            throw "gpuAlloc : Bad size for input argument #2: A scalar expected.";
        }
        columns = static_cast<int>(dtmp[0]);


#ifdef WITH_CUDA
        if (useCuda())
        {
            PointerCuda* dptrCuda = new PointerCuda(rows, columns, false);
            PointerManager::getInstance()->addGpuPointerInManager(dptrCuda);
            sciErr = createPointer(pvApiCtx, Rhs + 1, (void*)dptrCuda);
        }
#endif

#ifdef WITH_OPENCL
        if (!useCuda())
        {
            PointerOpenCL* dptrOpenCL = new PointerOpenCL(rows, columns, false);
            PointerManager::getInstance()->addGpuPointerInManager(dptrOpenCL);
            sciErr = createPointer(pvApiCtx, Rhs + 1, (void*)dptrOpenCL);
        }
#endif

        if (sciErr.iErr)
        {
            throw sciErr;
        }

        LhsVar(1) = Rhs + 1;
        PutLhsVar();
    }
    catch (const char* str)
    {
        Scierror(999, "%s: %s\n", fname, str);
    }
    catch (SciErr E)
    {
        printError(&E, 0);
    }
    return 0;
}
/* ========================================================================== */
