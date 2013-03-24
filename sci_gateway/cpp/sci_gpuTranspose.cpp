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

int sci_gpuTranspose(char *fname)
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
    bool    bComplex_A  = FALSE;
    int     inputType_A = 0;

    GpuPointer* gpuPtr = NULL;
    GpuPointer* gpuPtrResult = NULL;

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
            throw sciErr;
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
                throw "gpuTranspose : Bad type for input argument #1: Variables created with GPU functions expected.";
            }

            if (useCuda() && gpuPtr->getGpuType() != GpuPointer::CudaType)
            {
                throw "gpuTranspose : Bad type for input argument #2: A Cuda pointer expected.";
            }

            if (useCuda() == false && gpuPtr->getGpuType() != GpuPointer::OpenCLType)
            {
                throw "gpuTranspose : Bad type for input argument #2: A OpenCL pointer expected.";
            }
        }
        else if (inputType_A == sci_matrix)
        {
            // Get size and data
            if (isVarComplex(pvApiCtx, piAddr_A))
            {
                sciErr = getComplexMatrixOfDouble(pvApiCtx, piAddr_A, &rows, &cols, &h, &hi);
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtr = new PointerCuda(h, hi, rows, cols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    throw "gpuTranspose: not implemented with OpenCL.";
                }
#endif
            }
            else
            {
                sciErr = getMatrixOfDouble(pvApiCtx, piAddr_A, &rows, &cols, &h);
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtr = new PointerCuda(h, rows, cols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    throw "gpuTranspose: not implemented with OpenCL.";
                }
#endif
            }
            if (sciErr.iErr)
            {
                throw sciErr;
            }
        }
        else
        {
            throw "gpuTranspose : Bad type for input argument #1: A GPU or CPU matrix expected.";
        }

#ifdef WITH_OPENCL
        if (!useCuda())
        {
            throw "gpuTranspose: not implemented with OpenCL.";
        }
#endif

        // Performs operation
        gpuPtrResult = gpuPtr->transpose();

        // Put the result in scilab
        // Keep the result on the Device.
        PointerManager::getInstance()->addGpuPointerInManager(gpuPtrResult);
        sciErr = createPointer(pvApiCtx, Rhs + 1, (void*)gpuPtrResult);
        if (sciErr.iErr)
        {
            throw sciErr;
        }
        LhsVar(1) = Rhs + 1;

        if (inputType_A == 1 && gpuPtr != NULL)
        {
            delete gpuPtr;
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

    if (inputType_A == 1 && gpuPtr != NULL)
    {
        delete gpuPtr;
    }
    if (gpuPtrResult != NULL)
    {
        delete gpuPtrResult;
    }

    return EXIT_FAILURE;
}
/* ========================================================================== */
