/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
* Copyright (C) DIGITEO - 2011 - Cedric DELAMARRE
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
int sci_gpuFFT(char *fname)
{
    CheckRhs(1, 4);
    CheckLhs(1, 1);

    SciErr sciErr;

    int* piAddr_A       = NULL;
    double* h           = NULL;
    double* hi          = NULL;
    int iRows           = 0;
    int iCols           = 0;
    bool bComplex_A     = FALSE;
    int inputType       = 0;

    int iSign           = 0;
    int* piDim          = NULL;
    int iRowsDim        = 0;
    int iColsDim        = 0;
    int iSizeDim        = 0;

    int* piIncr         = NULL;
    //    int iRowsIncr       = 0;
    //    int iColsIncr       = 0;

    void* pvPtr         = NULL;

    double* pdblArg     = NULL;
    int iRowsArg        = 0;
    int iColsArg        = 0;
    int inputTypeArg    = 0;
    int* piAddrArg      = NULL;

    GpuPointer* gpuPtr = NULL;
    GpuPointer* gpuPtrResult = NULL;

    try
    {
        if (isGpuInit() == false)
        {
            throw "gpuFFT : gpu is not initialised. Please launch gpuInit() before use this function.";
        }

        // Get var env
        sciErr = getVarAddressFromPosition(pvApiCtx, 1, &piAddr_A);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        //check type
        sciErr = getVarType(pvApiCtx, piAddr_A, &inputType);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        if (inputType == sci_pointer)
        {
            sciErr = getPointer(pvApiCtx, piAddr_A, (void**)&pvPtr);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            gpuPtr = (GpuPointer*)pvPtr;
            if (!PointerManager::getInstance()->findGpuPointerInManager(gpuPtr))
            {
                throw "gpuFFT : Bad type for input argument #1. Only variables created with GPU functions allowed.";
            }

            if (useCuda() && gpuPtr->getGpuType() != GpuPointer::CudaType)
            {
                throw "gpuFFT : Bad type for input argument #1: A Cuda pointer expected.";
            }

            if (useCuda() == false && gpuPtr->getGpuType() != GpuPointer::OpenCLType)
            {
                throw "gpuFFT : Bad type for input argument #1: A OpenCL pointer expected.";
            }
        }
        else if (inputType == sci_matrix)
        {
            // Get size and data
            if (isVarComplex(pvApiCtx, piAddr_A))
            {
                sciErr = getComplexMatrixOfDouble(pvApiCtx, piAddr_A, &iRows, &iCols, &h, &hi);
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtr = new PointerCuda(h, hi, iRows, iCols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    Scierror(999, "gpuFFT: not implemented with OpenCL.\n");
                }
#endif
            }
            else
            {
                sciErr = getMatrixOfDouble(pvApiCtx, piAddr_A, &iRows, &iCols, &h);
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtr = new PointerCuda(h, iRows, iCols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    Scierror(999, "gpuFFT: not implemented with OpenCL.\n");
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
            throw "gpuFFT : Bad type for input argument #1 : A matrix or gpu pointer expected.";
        }

        for (int i = 2; i <= Rhs; i++)
        {
            sciErr = getVarAddressFromPosition(pvApiCtx, i, &piAddrArg);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            sciErr = getVarType(pvApiCtx, piAddrArg, &inputTypeArg);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            if (inputTypeArg != 1)
            {
                char str[70];
                sprintf(str, "gpuFFT : Bad type for input argument #%d : A real matrix expected.", i);
                throw str;
            }

            sciErr = getMatrixOfDouble(pvApiCtx, piAddrArg, &iRowsArg, &iColsArg, &pdblArg);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            if (iSign == 0)
            {
                if (iRowsArg * iColsArg != 1)
                {
                    char str[70];
                    sprintf(str, "gpuFFT : Bad size for input argument #%d : A scalar expected.", i);
                    throw str;
                }

                iSign = static_cast<int>(pdblArg[0]);

                if (iSign != 1 && iSign != -1)
                {
                    char str[70];
                    sprintf(str, "gpuFFT : Bad value for input argument #%d : 1 or -1 expected.", i);
                    throw str;
                }
            }
            else if (piDim == NULL)
            {
                iRowsDim = iRowsArg;
                iColsDim = iColsArg;
                iSizeDim = iRowsDim * iColsDim;

                if (iSizeDim > 3)
                {
                    char str[70];
                    sprintf(str, "gpuFFT : Bad size for input argument #%d : At most 3 expected.", i);
                    throw str;
                }

                piDim = new int[iSizeDim];
                for (int j = 0; j < iSizeDim; j++)
                {
                    piDim[j] = static_cast<int>(pdblArg[j]);
                }
            }
            else if (piIncr == NULL)
            {
                sciprint("WARNING : Argument #%d will not be used to compute the FFT.\n You can remove it.\n", i);

                //                iRowsIncr = iRowsArg;
                //                iColsIncr = iColsArg;
                //
                //                if(iRowsIncr != iRowsDim || iColsIncr != iColsDim)
                //                {
                //                    char str[70];
                //                    sprintf(str,"gpuFFT : Bad size for input argument #%d : Size of argument %d expected.", i, i-1);
                //                    throw str;
                //                }
                //
                //                piIncr = new int[iSizeDim];
                //                for(int j = 0; j < iSizeDim; j++)
                //                {
                //                    piIncr[j] = static_cast<int>(pdblArg[j]);
                //                }
            }
        }

        //        if(piDim && (piIncr == NULL))
        //        {
        //            throw "gpuFFT: Wrong number of input argument(s): 4 expected.\n";
        //        }

        if (Rhs == 1)
        {
            iSign = -1;
        }

        //Performs operation cufft
#ifdef WITH_OPENCL
        if (!useCuda())
        {
            throw "gpuFFT : not yet implemented with OpenCL.";
        }
#endif

        if (gpuPtr->getSize() == 1)
        {
            gpuPtrResult = gpuPtr->clone();
        }
        else
        {
            gpuPtrResult = gpuPtr->FFT(iSign, piDim, iSizeDim, piIncr);
        }

        // Keep the result on the Device.
        PointerManager::getInstance()->addGpuPointerInManager(gpuPtrResult);
        createPointer(pvApiCtx, Rhs + 1, (void*)gpuPtrResult);
        if (sciErr.iErr)
        {
            throw sciErr;
        }
        LhsVar(1) = Rhs + 1;

        // delete gpu pointer if input data was cpu pointer.
        if (inputType == 1)
        {
            delete gpuPtr;
            gpuPtr = NULL;
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

    if (inputType == 1 && gpuPtr != NULL)
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
