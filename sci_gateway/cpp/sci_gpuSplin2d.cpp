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
#include <string.h>
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
#include "gpuSplin2d.hxx"
/* ========================================================================== */
#include "api_scilab.h"
#include "Scierror.h"
#include "sciprint.h"
/* ========================================================================== */
int sci_gpuSplin2d(char *fname)
{
    CheckRhs(3, 4);
    CheckLhs(1, 1);

    SciErr sciErr;

    int* piAddr     = NULL;
    int iRows       = 0;
    int iCols       = 0;
    int inputType   = 0;
    double* h       = NULL;
    void* pvPtr     = NULL;

    bool isGpuPtr[3];

    SplineType spType = NOT_A_KNOT; // default

    int sizeOfX = 0;
    int sizeOfY = 0;
    int sizeOfZ = 0;
    int sizeOfC = 0;

    // input data
    std::vector<GpuPointer*> vectInputs;
    // output data
    GpuPointer* gpuPtrC = NULL;

    try
    {
        if (isGpuInit() == false)
        {
            throw "gpuSplin2d : gpu is not initialised. Please launch gpuInit() before use this function.";
        }

        // *** check type of input args and get it. ***
        for (int i = 0; i < 3; i++)
        {
            // Get var env
            sciErr = getVarAddressFromPosition(pvApiCtx, i + 1, &piAddr);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            //check type
            sciErr = getVarType(pvApiCtx, piAddr, &inputType);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            if (inputType == sci_pointer)
            {
                sciErr = getPointer(pvApiCtx, piAddr, (void**)&pvPtr);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }

                GpuPointer* gpuPtr = (GpuPointer*)pvPtr;
                if (!PointerManager::getInstance()->findGpuPointerInManager(gpuPtr))
                {
                    char str[100];
                    sprintf(str, "gpuSplin2d : Bad type for input argument #%d : Only variables created with GPU functions allowed.", i + 1);
                    throw str;
                }

                if (useCuda() && gpuPtr->getGpuType() != GpuPointer::CudaType)
                {
                    char str[100];
                    sprintf(str, "gpuSplin2d : Bad type for input argument #%d: A Cuda pointer expected.", i + 1);
                    throw str;
                }

                if (useCuda() == false && gpuPtr->getGpuType() != GpuPointer::OpenCLType)
                {
                    char str[100];
                    sprintf(str, "gpuSplin2d : Bad type for input argument #%d: A OpenCL pointer expected.", i + 1);
                    throw str;
                }

                if (gpuPtr->isGpuComplex())
                {
                    char str[100];
                    sprintf(str, "splin2d: Wrong type for input argument #%d : A real matrix expected.", i + 1);
                    throw str;
                }

                vectInputs.push_back(gpuPtr);
                isGpuPtr[i] = true;
            }
            else if (inputType == sci_matrix)
            {
                // Get size and data
                if (isVarComplex(pvApiCtx, piAddr))
                {
                    char str[100];
                    sprintf(str, "splin2d: Wrong type for input argument #%d : A real matrix expected.", i + 1);
                    throw str;
                }
                else
                {
                    sciErr = getMatrixOfDouble(pvApiCtx, piAddr, &iRows, &iCols, &h);
#ifdef WITH_CUDA
                    if (useCuda())
                    {
                        vectInputs.push_back(new PointerCuda(h, iRows, iCols));
                    }
#endif
#ifdef WITH_OPENCL
                    if (!useCuda())
                    {
                        Scierror(999, "splin2d: not implemented with OpenCL.\n");
                    }
#endif
                }

                if (sciErr.iErr)
                {
                    throw sciErr;
                }

                isGpuPtr[i] = false;
            }
            else
            {
                char str[100];
                sprintf(str, "splin2d: Bad type for input argument #%d : A matrix or gpu pointer expected.", i + 1);
                throw str;
            }
        }

        sizeOfX = vectInputs[0]->getSize();
        if (vectInputs[0]->getRows() != 1 || sizeOfX < 2)
        {
            throw "splin2d: Wrong size for input arguments #1: Rows vector of size >= 2 expected.";
        }

        sizeOfY = vectInputs[1]->getSize();
        if (vectInputs[1]->getRows() != 1 || sizeOfY < 2)
        {
            throw "splin2d: Wrong size for input arguments #2: Rows vector of size >= 2 expected.";
        }

        if (vectInputs[2]->getRows() != sizeOfX)
        {
            throw "splin2d: Wrong size for input arguments #3: Size of argument #1 expected in rows.";
        }

        if (vectInputs[2]->getCols() != sizeOfY)
        {
            throw "splin2d: Wrong size for input arguments #3: Size of argument #2 expected in columns.";
        }

        // out mode
        if (Rhs == 4)
        {
            // Get var env
            sciErr = getVarAddressFromPosition(pvApiCtx, 4, &piAddr);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            //check type
            sciErr = getVarType(pvApiCtx, piAddr, &inputType);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            if (inputType != sci_strings)
            {
                throw "splin2d: Wrong type for input argument #4: A String expected.";
            }

            //first call to retrieve dimensions
            sciErr = getMatrixOfString(pvApiCtx, piAddr, &iRows, &iCols, NULL, NULL);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            if (iRows * iCols != 1)
            {
                throw "splin2d: Wrong size for input argument #4: A scalar String expected.";
            }

            int iLen = 0;
            //second call to retrieve length of each string
            sciErr = getMatrixOfString(pvApiCtx, piAddr, &iRows, &iCols, &iLen, NULL);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            char* pstData = new char[iLen + 1]; //+ 1 for null termination
            //third call to retrieve data
            sciErr = getMatrixOfString(pvApiCtx, piAddr, &iRows, &iCols, &iLen, &pstData);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            if (strcmp(pstData, "not_a_knot") == 0)
            {
                spType = NOT_A_KNOT;
            }
            else if (strcmp(pstData, "monotone") == 0)
            {
                spType = MONOTONE;
            }
            else if (strcmp(pstData, "natural") == 0)
            {
                spType = NATURAL;
            }
            else if (strcmp(pstData, "periodic") == 0)
            {
                spType = PERIODIC;
            }
            else if (strcmp(pstData, "fast") == 0)
            {
                spType = FAST;
            }
            else if (strcmp(pstData, "fast_periodic") == 0)
            {
                spType = FAST_PERIODIC;
            }
            else // undefined type ans CLAMPED
            {
                char str[100];
                sprintf(str, "gpuSplin2d : Wrong value for input argument #4 : Unsupported '%s' type.", pstData);
                throw str;
            }

            delete pstData;
        }

        // *** Perform operation. ***
#ifdef WITH_OPENCL
        if (!useCuda())
        {
            Scierror(999, "splin2d: not implemented with OpenCL.\n");
        }
#endif

        gpuPtrC = gpuSplin2d(vectInputs[0], vectInputs[1], vectInputs[2], spType);

        if (gpuPtrC == NULL)
        {
            throw "gpuSplin2d : Strict increasing vector expected.";
        }

        // *** Return result in Scilab. ***
        // Keep the result on the Device.
        PointerManager::getInstance()->addGpuPointerInManager(gpuPtrC);
        createPointer(pvApiCtx, Rhs + 1, (void*)gpuPtrC);
        if (sciErr.iErr)
        {
            throw sciErr;
        }
        LhsVar(1) = Rhs + 1;

        for (int i = 0; i < vectInputs.size(); i++)
        {
            if (!isGpuPtr[i])
            {
                delete vectInputs[i];
            }
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

    // input
    for (int i = 0; i < vectInputs.size(); i++)
    {
        if (!isGpuPtr[i])
        {
            delete vectInputs[i];
        }
    }

    // output
    if (gpuPtrC)
    {
        delete gpuPtrC;
    }

    return EXIT_FAILURE;
}
/* ========================================================================== */
