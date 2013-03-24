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
/* ========================================================================== */
#include "api_scilab.h"
#include "Scierror.h"
#include "sciprint.h"
/* ========================================================================== */

int sci_gpuMatrix(char *fname)
{
    CheckRhs(2, 3);
    CheckLhs(1, 1);

    SciErr sciErr;

    int*    piAddr_A    = NULL;
    int     inputType_A = 0;
    int*    piAddr_R    = NULL;
    int     inputType_R = 0;
    int*    piAddr_C    = NULL;
    int     inputType_C = 0;

    int     rows        = 0;
    int     cols        = 0;
    int     newRows     = 0;
    int     newCols     = 0;

    void*   pvPtr       = NULL;
    GpuPointer* gpuPtrA = NULL;

    try
    {
        if (!isGpuInit())
        {
            throw "gpu is not initialised. Please launch gpuInit() before use this function.";
        }

        //--- Get input matrix ---
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

        //--- Get new Rows size or vector of sizes---
        sciErr = getVarAddressFromPosition(pvApiCtx, 2, &piAddr_R);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        // Get size and data
        sciErr = getVarType(pvApiCtx, piAddr_R, &inputType_R);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        if (inputType_R != sci_matrix)
        {
            throw "gpuMatrix : Bad type for input argument #2: A real scalar or row vector expected.";
        }

        if (isVarComplex(pvApiCtx, piAddr_A))
        {
            throw "gpuMatrix : Bad type for input argument #2: A real scalar or row vector expected.";
        }
        else
        {
            double* dRows = NULL;
            sciErr = getMatrixOfDouble(pvApiCtx, piAddr_R, &rows, &cols, &dRows);
            if (sciErr.iErr)
            {
                throw sciErr;
            }
            if (nbInputArgument(pvApiCtx) == 2)
            {
                if (rows != 1 || cols != 2)
                {
                    throw "gpuMatrix : Bad size for input argument #2: A row vector of size two expected.";
                }

                newRows = (int)dRows[0];
                newCols = (int)dRows[1];

                if (newCols < -1 || newCols == 0)
                {
                    throw "gpuMatrix : Wrong value for input argument #3: -1 or positive value expected.";
                }
            }
            else
            {
                newRows = (int)(*dRows);
            }

            if (newRows < -1 || newRows == 0)
            {
                throw "gpuMatrix : Wrong value for input argument #2: -1 or positive value expected.";
            }
        }

        if (nbInputArgument(pvApiCtx) == 3)
        {
            //--- Get new Cols size---
            sciErr = getVarAddressFromPosition(pvApiCtx, 3, &piAddr_C);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            // Get size and data
            sciErr = getVarType(pvApiCtx, piAddr_C, &inputType_C);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            if (inputType_C != sci_matrix)
            {
                throw "gpuMatrix : Bad type for input argument #3: A real scalar expected.";
            }

            if (isVarComplex(pvApiCtx, piAddr_A))
            {
                throw "gpuMatrix : Bad type for input argument #3: A real scalar expected.";
            }
            else
            {
                double* dCols = NULL;
                sciErr = getMatrixOfDouble(pvApiCtx, piAddr_C, &rows, &cols, &dCols);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }

                newCols = (int)(*dCols);

                if (newCols < -1 || newCols == 0)
                {
                    throw "gpuMatrix : Wrong value for input argument #3: -1 or positive value expected.";
                }
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
                throw "gpuMatrix : Bad type for input argument #1: Variables created with GPU functions expected.";
            }

            if (useCuda() && gpuPtrA->getGpuType() != GpuPointer::CudaType)
            {
                throw "gpuMatrix : Bad type for input argument #1: A Cuda pointer expected.";
            }

            if (useCuda() == false && gpuPtrA->getGpuType() != GpuPointer::OpenCLType)
            {
                throw "gpuMatrix : Bad type for input argument #1: A OpenCL pointer expected.";
            }

            rows = gpuPtrA->getRows();
            cols = gpuPtrA->getCols();
        }
        else if (inputType_A == sci_matrix)
        {
            double* h = NULL;
            if (isVarComplex(pvApiCtx, piAddr_A))
            {
                double* hi = NULL;
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
                    Scierror(999, "gpuMatrix: not implemented with OpenCL.\n");
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
                    Scierror(999, "gpuMatrix: not implemented with OpenCL.\n");
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
            throw "gpuMatrix : Bad type for input argument #1: A GPU or CPU matrix expected.";
        }

        if (newRows == -1 && newCols != -1)
        {
            newRows = rows * cols / newCols;
        }
        else if (newRows != -1 && newCols == -1)
        {
            newCols = rows * cols / newRows;
        }

        if (rows * cols != newRows * newCols)
        {
            throw "gpuMatrix : Wrong value for input arguments #2 and 3: Correct size expected.";
        }

#ifdef WITH_OPENCL
        if (!useCuda())
        {
            Scierror(999, "gpuMatrix: not implemented with OpenCL.\n");
        }
#endif

        GpuPointer* gpuOut = gpuPtrA->clone();
        gpuOut->setRows(newRows);
        gpuOut->setCols(newCols);

        // Put the result in scilab
        PointerManager::getInstance()->addGpuPointerInManager(gpuOut);
        sciErr = createPointer(pvApiCtx, Rhs + 1, (void*)gpuOut);
        LhsVar(1) = Rhs + 1;

        if (inputType_A == 1 && gpuPtrA != NULL)
        {
            delete gpuPtrA;
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

    return EXIT_FAILURE;
}
/* ========================================================================== */
