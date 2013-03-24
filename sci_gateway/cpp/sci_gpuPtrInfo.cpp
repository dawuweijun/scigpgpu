/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
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
#include "gw_gpu.h"
#include "gpuContext.hxx"
/* ========================================================================== */
#include "api_scilab.h"
#include "Scierror.h"
#include "sciprint.h"
/* ========================================================================== */

int sci_gpuPtrInfo(char *fname)
{
    CheckRhs(1, 1);
    CheckLhs(1, 1);

    int*        piAddr_A    = NULL;
    void*       pvPtr       = NULL;
    GpuPointer* gpuPtr      = NULL;
    int         inputType_A;
    SciErr      sciErr;
    char*       pstCuda     = "Cuda Pointer";
    char*       pstOpenCL   = "OpenCL Pointer";

    try
    {
        sciErr = getVarAddressFromPosition(pvApiCtx, 1, &piAddr_A);
        if (sciErr.iErr)
        {
            throw sciErr;
        }
        sciErr = getVarType(pvApiCtx, piAddr_A, &inputType_A);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        if (inputType_A == sci_pointer)
        {
            sciErr = getPointer(pvApiCtx, piAddr_A, (void**)&pvPtr);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            gpuPtr = (GpuPointer*)pvPtr;
            if (!PointerManager::getInstance()->findGpuPointerInManager(gpuPtr))
            {
                throw "gpuPtrInfo : Bad type for input argument #1: Variables created with GPU functions expected.";
            }

            if (gpuPtr->getGpuType() == GpuPointer::CudaType)
            {
                sciErr = createMatrixOfString(pvApiCtx, Rhs + 1, 1, 1, &pstCuda);
            }
            else if (gpuPtr->getGpuType() == GpuPointer::OpenCLType)
            {
                sciErr = createMatrixOfString(pvApiCtx, Rhs + 1, 1, 1, &pstOpenCL);
            }
        }
        else
        {
            throw "gpuPtrInfo : Bad type for input argument #1: A GPU matrix expected.";
        }


        if (sciErr.iErr)
        {
            throw sciErr;
        }

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
    return EXIT_FAILURE;
}
