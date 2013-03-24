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

int sci_gpuSum(char *fname)
{
    CheckRhs(1, 1);
    CheckLhs(1, 1);

    SciErr  sciErr;

    int*    piAddr_A    = NULL;
    bool    bComplex_A  = FALSE;
    int     inputType_A = 0;
    double* h           = NULL;
    double* hi          = NULL;
    int     rows        = 0;
    int     cols        = 0;

    void*   pvPtr       = NULL;

    double*  res        = 0;

    GpuPointer* gpuPtr = NULL;

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
                throw "gpuSum : Bad type for input argument #1: Variables created with GPU functions expected.";
            }

            if (useCuda() && gpuPtr->getGpuType() != GpuPointer::CudaType)
            {
                throw "gpuSum : Bad type for input argument #1: A Cuda pointer expected.";
            }

            if (useCuda() == false && gpuPtr->getGpuType() != GpuPointer::OpenCLType)
            {
                throw "gpuSum : Bad type for input argument #1: A OpenCL pointer expected.";
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
                    throw "not implemented with OpenCL.";
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
                    throw "not implemented with OpenCL.";
                }
#endif
            }
        }
        else
        {
            throw "gpuSum : Bad type for input argument #1: A GPU or CPU matrix expected.";
        }

#ifdef WITH_CUDA
        if (useCuda())
        {
            // Keep the result on the Host.
            // Put the result in scilab
            if (!gpuPtr->isGpuComplex())
            {
                double res = gpuPtr->getSum();
                sciErr = createMatrixOfDouble(pvApiCtx, Rhs + 1, 1, 1, &res);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }
            }
            else
            {
                cuDoubleComplex res = ((PointerCuda*)gpuPtr)->getComplexSum();
                sciErr = createComplexMatrixOfDouble(pvApiCtx, Rhs + 1, 1, 1, &res.x, &res.y);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }
            }

            if (inputType_A == 1 && gpuPtr != NULL)
            {
                delete gpuPtr;
            }

            LhsVar(1) = Rhs + 1;
        }
#endif

#ifdef WITH_OPENCL
        if (!useCuda())
        {
            throw "not implemented with OpenCL.";
        }
#endif

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

