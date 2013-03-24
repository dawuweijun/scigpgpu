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

int sci_gpuOnes(char *fname)
{
    CheckLhs(1, 1);

    void* pvPtr = NULL;
    int* piAddr = NULL;
    SciErr sciErr;
    int inputType;

    int iRows = 0;
    int iCols = 0;

    GpuPointer* gpOut = NULL;

    try
    {
        if (!isGpuInit())
        {
            throw "gpu is not initialised. Please launch gpuInit() before use this function.";
        }

        if (Rhs == 1)
        {
            sciErr = getVarAddressFromPosition(pvApiCtx, 1, &piAddr);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            sciErr = getVarType(pvApiCtx, piAddr, &inputType);
            if (inputType == sci_pointer)
            {
                sciErr = getPointer(pvApiCtx, piAddr, (void**)&pvPtr);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }

                GpuPointer* gmat = (GpuPointer*)(pvPtr);
                if (!PointerManager::getInstance()->findGpuPointerInManager(gmat))
                {
                    throw "gpuOnes : Bad type for input argument #1. Only variables created with GPU functions allowed.";
                }

                if (useCuda() && gmat->getGpuType() != GpuPointer::CudaType)
                {
                    throw "gpuOnes : Bad type for input argument #1: A Cuda pointer expected.";
                }

                if (useCuda() == false && gmat->getGpuType() != GpuPointer::OpenCLType)
                {
                    throw "gpuOnes : Bad type for input argument #1: A OpenCL pointer expected.";
                }

                if (gmat->getDims() > 2)
                {
                    throw "gpuOnes : Hypermatrix not yet implemented.";
                }

                iRows = gmat->getRows();
                iCols = gmat->getCols();
            }
            else if (inputType == sci_matrix)
            {
                // Get size and data
                double* h;
                sciErr = getMatrixOfDouble(pvApiCtx, piAddr, &iRows, &iCols, &h);
            }
            else
            {
                throw "gpuOnes : Bad type for input argument #1 : A Matrix or GPU pointer expected.";
            }
        }
        else
        {
            if (Rhs > 2)
            {
                throw "gpuOnes : Hypermatrix not yet implemented.";
            }

            int* piDimsArray = new int[Rhs];
            for (int i = 0; i < Rhs; i++)
            {
                sciErr = getVarAddressFromPosition(pvApiCtx, i + 1, &piAddr);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }

                sciErr = getVarType(pvApiCtx, piAddr, &inputType);
                if (inputType != sci_matrix)
                {
                    throw "gpuOnes : Bad type for input argument #%d : A Matrix expected.";
                }

                double* h;
                sciErr = getMatrixOfDouble(pvApiCtx, piAddr, &iRows, &iCols, &h);
                if (iRows * iCols != 1)
                {
                    char str[100];
                    sprintf(str, "gpuOnes : Wrong size for input argument #%d : A scalar expected.", i + 1);
                    throw str;
                }

                piDimsArray[i] = (int)h[0];
            }

            iRows = piDimsArray[0];
            iCols = piDimsArray[1];

            delete piDimsArray;
        }

#ifdef WITH_CUDA
        if (useCuda())
        {
            gpOut = new PointerCuda(iRows, iCols, false);
            gpOut->initMatrix(1);
        }
#endif
#ifdef WITH_OPENCL
        if (!useCuda())
        {
            Scierror(999, "gpuOnes: not implemented with OpenCL.\n");
        }
#endif

        PointerManager::getInstance()->addGpuPointerInManager(gpOut);
        sciErr = createPointer(pvApiCtx, Rhs + 1, (void*)gpOut);
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
