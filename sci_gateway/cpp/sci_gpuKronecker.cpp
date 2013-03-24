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
#include "gpuKronecker.hxx"
/* ========================================================================== */
#include "api_scilab.h"
#include "Scierror.h"
#include "sciprint.h"
/* ========================================================================== */

int sci_gpuKronecker(char *fname)
{
    CheckRhs(2, 2);
    CheckLhs(1, 1);

    SciErr sciErr;

    int*    piAddr_A    = NULL;
    int*    piAddr_B    = NULL;

    GpuPointer* gpuPtrA = NULL;
    GpuPointer* gpuPtrB = NULL;
    GpuPointer* gpuPtrC = NULL;

    double* h           = NULL;
    double* hi          = NULL;
    int rows            = 0;
    int cols            = 0;

    void* pvPtrA        = NULL;
    void* pvPtrB        = NULL;

    int inputType_A;
    int inputType_B;

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

        /* ---- Check type of arguments and get data ---- */
        /*                                                */
        /*  Pointer to host / Pointer to device           */
        /*  Matrix real / Matrix complex                  */
        /*                                                */
        /* ---------------------------------------------- */

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

        if (inputType_A == sci_pointer)
        {
            sciErr = getPointer(pvApiCtx, piAddr_A, (void**)&pvPtrA);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            gpuPtrA = (GpuPointer*)pvPtrA;
            if (!PointerManager::getInstance()->findGpuPointerInManager(gpuPtrA))
            {
                throw "gpuKronecker : Bad type for input argument #1: Variables created with GPU functions expected.";
            }

            if (useCuda() && gpuPtrA->getGpuType() != GpuPointer::CudaType)
            {
                throw "gpuKronecker : Bad type for input argument #1: A Cuda pointer expected.";
            }

            if (useCuda() == false && gpuPtrA->getGpuType() != GpuPointer::OpenCLType)
            {
                throw "gpuKronecker : Bad type for input argument #1: A OpenCL pointer expected.";
            }
        }
        else if (inputType_A == sci_matrix)
        {
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
                    gpuPtrA = new PointerCuda(h, hi, rows, cols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    throw "gpuKronecker: not implemented with OpenCL.";
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
                    gpuPtrA = new PointerCuda(h, rows, cols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    throw "gpuKronecker: not implemented with OpenCL.";
                }
#endif
            }
        }
        else
        {
            throw "gpuKronecker : Bad type for input argument #1: A GPU or CPU matrix expected.";
        }

        if (inputType_B == sci_pointer)
        {
            sciErr = getPointer(pvApiCtx, piAddr_B, (void**)&pvPtrB);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            gpuPtrB = (GpuPointer*)pvPtrB;
            if (!PointerManager::getInstance()->findGpuPointerInManager(gpuPtrB))
            {
                throw "gpuKronecker : Bad type for input argument #2: Variables created with GPU functions expected.";
            }

            if (useCuda() && gpuPtrB->getGpuType() != GpuPointer::CudaType)
            {
                throw "gpuKronecker : Bad type for input argument #2: A Cuda pointer expected.";
            }

            if (useCuda() == false && gpuPtrB->getGpuType() != GpuPointer::OpenCLType)
            {
                throw "gpuKronecker : Bad type for input argument #2: A OpenCL pointer expected.";
            }
        }
        else if (inputType_B == sci_matrix)
        {
            if (isVarComplex(pvApiCtx, piAddr_B))
            {
                sciErr = getComplexMatrixOfDouble(pvApiCtx, piAddr_B, &rows, &cols, &h, &hi);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtrB = new PointerCuda(h, hi, rows, cols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    Scierror(999, "gpuKronecker: not implemented with OpenCL.\n");
                }
#endif
            }
            else
            {
                sciErr = getMatrixOfDouble(pvApiCtx, piAddr_B, &rows, &cols, &h);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtrB = new PointerCuda(h, rows, cols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    Scierror(999, "gpuKronecker: not implemented with OpenCL.\n");
                }
#endif
            }
        }
        else
        {
            throw "gpuKronecker : Bad type for input argument #2: A GPU or CPU matrix expected.";
        }

#ifdef WITH_OPENCL
        if (!useCuda())
        {
            throw "gpuKronecker: not implemented with OpenCL.";
        }
#endif

        //performe operation.
        gpuPtrC = gpuKronecker(gpuPtrA, gpuPtrB);

        // Keep the result on the Device.
        PointerManager::getInstance()->addGpuPointerInManager(gpuPtrC);
        sciErr = createPointer(pvApiCtx, Rhs + 1, (void*)gpuPtrC);
        if (sciErr.iErr)
        {
            throw sciErr;
        }
        LhsVar(1) = Rhs + 1;

        if (inputType_A == sci_matrix && gpuPtrA != NULL)
        {
            delete gpuPtrA;
        }
        if (inputType_B == sci_matrix && gpuPtrB != NULL)
        {
            delete gpuPtrB;
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

    if (useCuda())
    {
        if (inputType_A == sci_matrix && gpuPtrA != NULL)
        {
            delete gpuPtrA;
        }
        if (inputType_B == sci_matrix && gpuPtrB != NULL)
        {
            delete gpuPtrB;
        }
        if (gpuPtrC != NULL)
        {
            delete gpuPtrC;
        }
    }

    return EXIT_FAILURE;
}
/* ========================================================================== */
