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

/* ========================================================================== */
int sci_gpuSetData(char *fname)
{
    CheckLhs(1, 1);
    CheckRhs(1, 1);

    double* h = NULL;
    double* hi = NULL;
    int rows = 0, columns = 0;
    int* piAddr = NULL;
#ifdef WITH_CUDA
    cublasStatus status;
#endif
    SciErr sciErr;
    int inputType_A;

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
        sciErr = getVarType(pvApiCtx, piAddr, &inputType_A);
        if (sciErr.iErr)
        {
            throw sciErr;
        }
        if (inputType_A != sci_matrix)
        {
            throw "gpuSetData : Bad type for input argument #1 : A matrix expected.";
        }

        GpuPointer* dptr;
        if (isVarComplex(pvApiCtx, piAddr))
        {
            sciErr = getComplexMatrixOfDouble(pvApiCtx, piAddr, &rows, &columns, &h, &hi);
            if (sciErr.iErr)
            {
                throw sciErr;
            }
#ifdef WITH_CUDA
            if (useCuda())
            {
                dptr = new PointerCuda(h, hi, rows, columns);
            }
#endif
#ifdef WITH_OPENCL
            if (!useCuda())
            {
                Scierror(999, "gpuSetData: Complex argument not implemented with OpenCL.\n");
            }
#endif
        }
        else
        {
            sciErr = getMatrixOfDouble(pvApiCtx, piAddr, &rows, &columns, &h);
            if (sciErr.iErr)
            {
                throw sciErr;
            }
#ifdef WITH_CUDA
            if (useCuda())
            {
                dptr = new PointerCuda(h, rows, columns);
            }
#endif
#ifdef WITH_OPENCL
            if (!useCuda())
            {
                dptr = new PointerOpenCL(h, rows, columns);
            }
#endif
        }

        PointerManager::getInstance()->addGpuPointerInManager(dptr);
        sciErr = createPointer(pvApiCtx, Rhs + 1, (void*)dptr);
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
