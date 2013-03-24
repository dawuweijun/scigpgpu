/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
* Copyright (C) DIGITEO - 2010-2012 - Cedric DELAMARRE
* Copyright (C) Scilab Enterprises - 2013 - Cedric DELAMARRE
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
#include "cudaElemMin.hxx"
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

int sci_gpuMin(char *fname)
{
    CheckRhs(1, 2);
    CheckLhs(1, 1);

    SciErr sciErr;

    int*    piAddr_A    = NULL;
    int     inputType_A = 0;
    int*    piAddr_B    = NULL;
    int     inputType_B = 0;
    double* h           = NULL;
    double* hi          = NULL;
    int     rows        = 0;
    int     cols        = 0;
    int     rowsB       = 0;
    int     colsB       = 0;

    void*   pvPtr       = NULL;

    GpuPointer* gpuPtrA = NULL;
    GpuPointer* gpuPtrB = NULL;

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

        if (nbInputArgument(pvApiCtx) == 2)
        {
            // Get var env
            sciErr = getVarAddressFromPosition(pvApiCtx, 2, &piAddr_B);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            // Get size and data
            sciErr = getVarType(pvApiCtx, piAddr_B, &inputType_B);
            if (sciErr.iErr)
            {
                throw sciErr;
            }
        }


        if (inputType_A == sci_pointer)
        {
            sciErr = getPointer(pvApiCtx, piAddr_A, (void**)&pvPtr);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            gpuPtrA = (GpuPointer*)pvPtr;
            if (!PointerManager::getInstance()->findGpuPointerInManager(gpuPtrA))
            {
                throw "gpuMin : Bad type for input argument #1: Variables created with GPU functions expected.";
            }

            if (useCuda() && gpuPtrA->getGpuType() != GpuPointer::CudaType)
            {
                throw "gpuMin : Bad type for input argument #1: A Cuda pointer expected.";
            }

            if (useCuda() == false && gpuPtrA->getGpuType() != GpuPointer::OpenCLType)
            {
                throw "gpuMin : Bad type for input argument #1: A OpenCL pointer expected.";
            }

            rows = gpuPtrA->getRows();
            cols = gpuPtrA->getCols();
        }
        else if (inputType_A == sci_matrix)
        {
            if (isVarComplex(pvApiCtx, piAddr_A))
            {
                sciErr = getComplexMatrixOfDouble(pvApiCtx, piAddr_A, &rows, &cols, &h, &hi);
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtrA = new PointerCuda(h, hi, rows, cols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    Scierror(999, "gpuMin: not implemented with OpenCL.\n");
                }
#endif
            }
            else
            {
                sciErr = getMatrixOfDouble(pvApiCtx, piAddr_A, &rows, &cols, &h);
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtrA = new PointerCuda(h, rows, cols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    Scierror(999, "gpuMin: not implemented with OpenCL.\n");
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
            throw "gpuMin : Bad type for input argument #1: A GPU or CPU matrix expected.";
        }

        if (nbInputArgument(pvApiCtx) == 2)
        {
            if (inputType_B == sci_pointer)
            {
                sciErr = getPointer(pvApiCtx, piAddr_B, (void**)&pvPtr);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }

                gpuPtrB = (GpuPointer*)pvPtr;
                if (!PointerManager::getInstance()->findGpuPointerInManager(gpuPtrB))
                {
                    throw "gpuMin : Bad type for input argument #2: Variables created with GPU functions expected.";
                }

                if (useCuda() && gpuPtrB->getGpuType() != GpuPointer::CudaType)
                {
                    throw "gpuMin : Bad type for input argument #2: A Cuda pointer expected.";
                }

                if (useCuda() == false && gpuPtrB->getGpuType() != GpuPointer::OpenCLType)
                {
                    throw "gpuMin : Bad type for input argument #2: A OpenCL pointer expected.";
                }

                if (gpuPtrB->getRows() != rows || gpuPtrB->getCols() != cols)
                {
                    throw "gpuMin : Bad type for input argument #2: Same size expected.";
                }
            }
            else if (inputType_B == sci_matrix)
            {
                if (isVarComplex(pvApiCtx, piAddr_B))
                {
                    sciErr = getComplexMatrixOfDouble(pvApiCtx, piAddr_B, &rowsB, &colsB, &h, &hi);
#ifdef WITH_CUDA
                    if (useCuda())
                    {
                        gpuPtrB = new PointerCuda(h, hi, rows, cols);
                    }
#endif
#ifdef WITH_OPENCL
                    if (!useCuda())
                    {
                        Scierror(999, "gpuMin: not implemented with OpenCL.\n");
                    }
#endif
                }
                else
                {
                    sciErr = getMatrixOfDouble(pvApiCtx, piAddr_B, &rowsB, &colsB, &h);
#ifdef WITH_CUDA
                    if (useCuda())
                    {
                        gpuPtrB = new PointerCuda(h, rows, cols);
                    }
#endif
#ifdef WITH_OPENCL
                    if (!useCuda())
                    {
                        Scierror(999, "gpuMin: not implemented with OpenCL.\n");
                    }
#endif
                }

                if (rowsB != rows || colsB != cols)
                {
                    throw "gpuMin : Bad type for input argument #2: Same size expected.";
                }

                if (sciErr.iErr)
                {
                    throw sciErr;
                }
            }
            else
            {
                throw "gpuMin : Bad type for input argument #2: A GPU or CPU matrix expected.";
            }
        }

#ifdef WITH_OPENCL
        if (!useCuda())
        {
            Scierror(999, "gpuMin: not implemented with OpenCL.\n");
        }
#endif

        if (gpuPtrB == NULL)
        {
            if (gpuPtrA->isGpuComplex())
            {
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    throw "gpuMin: not implemented with OpenCL.";
                }
#endif

                sciprint("To find the min value of complex matrix, the function performs operation abs(real)+abs(imaginary).\n");
#ifdef WITH_CUDA
                if (useCuda())
                {
                    // Performs operation using cublas
                    cuDoubleComplex res = ((PointerCuda*)gpuPtrA)->getComplexMin();

                    // Put the result in scilab
                    sciErr = createComplexMatrixOfDouble(pvApiCtx, Rhs + 1, 1, 1, &res.x, &res.y);
                }
#endif
            }
            else
            {
                // Performs operation
                double res = gpuPtrA->getMin();

                // Put the result in scilab
                sciErr = createMatrixOfDouble(pvApiCtx, Rhs + 1, 1, 1, &res);
            }
        }
        else
        {
            GpuPointer* gpuPtrRes = NULL;
            if (gpuPtrA->isGpuComplex() || gpuPtrB->isGpuComplex())
            {
                sciprint("To find the min value of complex matrix, the function performs operation abs(real)+abs(imaginary).\n");
            }

#ifdef WITH_CUDA
            if (useCuda())
            {
                gpuPtrRes = new PointerCuda(rows, cols, gpuPtrA->isGpuComplex() || gpuPtrB->isGpuComplex());

                // Performs operation
                cudaElemMin(dynamic_cast<PointerCuda*>(gpuPtrA), dynamic_cast<PointerCuda*>(gpuPtrB), dynamic_cast<PointerCuda*>(gpuPtrRes));
            }
#endif

            // Put the result in scilab
            PointerManager::getInstance()->addGpuPointerInManager(gpuPtrRes);
            sciErr = createPointer(pvApiCtx, Rhs + 1, (void*)gpuPtrRes);
        }

        if (sciErr.iErr)
        {
            throw sciErr;
        }

        LhsVar(1) = Rhs + 1;

        if (inputType_A == 1 && gpuPtrA != NULL)
        {
            delete gpuPtrA;
        }
        if (inputType_B == 1 && gpuPtrB != NULL)
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

    if (inputType_A == 1 && gpuPtrA != NULL)
    {
        delete gpuPtrA;
    }
    if (inputType_B == 1 && gpuPtrB != NULL)
    {
        delete gpuPtrB;
    }

    return EXIT_FAILURE;
}
/* ========================================================================== */
