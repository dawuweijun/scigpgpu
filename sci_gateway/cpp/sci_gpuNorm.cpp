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

int sci_gpuNorm(char *fname)
{
    CheckRhs(1, 1);
    CheckLhs(1, 1);

    SciErr sciErr;

    int*    piAddr_A    = NULL;
    double* h           = NULL;
    double* hi          = NULL;
    int     rows        = 0;
    int     cols        = 0;

    void*   pvPtr       = NULL;

    int     inputType_A = 0;

    GpuPointer* gpuPtr = NULL;

    double result       = 0;

    try
    {
        if (!isGpuInit())
        {
            throw "gpu is not initialised. Please launch gpuInit() before use this function.";
        }

        // Get var env
        sciErr = getVarAddressFromPosition(pvApiCtx, 1, &piAddr_A);
        if (sciErr.iErr)
        {
            printError(&sciErr, 0);
            return EXIT_FAILURE;
        }

        // Get size and data
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
                throw "gpuNorm : Bad type for input argument #1: Variables created with GPU functions expected.";
            }

            if (useCuda() && gpuPtr->getGpuType() != GpuPointer::CudaType)
            {
                throw "gpuNorm : Bad type for input argument #1: A Cuda pointer expected.";
            }

            if (useCuda() == false && gpuPtr->getGpuType() != GpuPointer::OpenCLType)
            {
                throw "gpuNorm : Bad type for input argument #1: A OpenCL pointer expected.";
            }
        }
        else if (inputType_A == sci_matrix)
        {
            // Get size and data
            if (isVarComplex(pvApiCtx, piAddr_A))
            {
                sciErr = getComplexMatrixOfDouble(pvApiCtx, piAddr_A, &rows, &cols, &h, &hi);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtr = new PointerCuda(h, hi, rows, cols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    throw "gpuNorm: not implemented with OpenCL.";
                }
#endif
            }
            else
            {
                sciErr = getMatrixOfDouble(pvApiCtx, piAddr_A, &rows, &cols, &h);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtr = new PointerCuda(h, rows, cols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    throw "gpuNorm: not implemented with OpenCL.";
                }
#endif
            }
        }
        else
        {
            throw "gpuNorm : Bad type for input argument #1: A GPU or CPU matrix expected.";
        }

#ifdef WITH_CUDA
        if (useCuda())
        {
            result = gpuPtr->getNorm();
        }
#endif
#ifdef WITH_OPENCL
        if (!useCuda())
        {
            throw "gpuNorm: not implemented with OpenCL.";
        }
#endif

        // Keep the result on the Host.
        // Put the result in scilab
        sciErr = createMatrixOfDouble(pvApiCtx, Rhs + 1, 1, 1, &result);
        if (sciErr.iErr)
        {
            throw sciErr;
        }
        LhsVar(1) = Rhs + 1;

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

    if (inputType_A == 1 && gpuPtr != NULL)
    {
        delete gpuPtr;
    }

    return EXIT_FAILURE;
}
/* ========================================================================== */
