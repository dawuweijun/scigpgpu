/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2012 - Cedric DELAMARRE
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
#include "interp.h"
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
int sci_gpuInterp(char *fname)
{
    CheckRhs(4, 5);
    CheckLhs(1, 4);

    SciErr sciErr;

    int* piAddr     = NULL;
    int iRows       = 0;
    int iCols       = 0;
    int inputType   = 0;
    double* h       = NULL;
    void* pvPtr     = NULL;

    bool isGpuPtr[4];

    int iType       = 8; // default
    int sizeOfXp    = 0;
    int sizeOfX     = 0;

    // input data
    std::vector<GpuPointer*> vectInputs;

    // output data
    GpuPointer* tabOutputs[4];
    tabOutputs[0] = NULL;

    try
    {
        if (isGpuInit() == false)
        {
            throw "gpuInterp : gpu is not initialised. Please launch gpuInit() before use this function.";
        }

        // *** check type of input args and get it. ***
        for (int i = 0; i < 4; i++)
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
                    sprintf(str, "gpuInterp : Bad type for input argument #%d : Only variables created with GPU functions allowed.", i + 1);
                    throw str;
                }

                if (useCuda() && gpuPtr->getGpuType() != GpuPointer::CudaType)
                {
                    throw "gpuInterp : Bad type for input argument #2: A Cuda pointer expected.";
                }

                if (useCuda() == false && gpuPtr->getGpuType() != GpuPointer::OpenCLType)
                {
                    throw "gpuInterp : Bad type for input argument #2: A OpenCL pointer expected.";
                }

                if (gpuPtr->isGpuComplex())
                {
                    char str[100];
                    sprintf(str, "gpuInterp : Wrong type for input argument #%d : A real matrix expected.", i + 1);
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
                    sprintf(str, "gpuInterp : Wrong type for input argument #%d : A real matrix expected.", i + 1);
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
                        Scierror(999, "gpuInterp: not implemented with OpenCL.\n");
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
                sprintf(str, "gpuInterp : Bad type for input argument #%d : A matrix or gpu pointer expected.", i + 1);
                throw str;
            }
        }

        sizeOfXp = vectInputs[0]->getSize();
        sizeOfX  = vectInputs[1]->getSize();

        if ((vectInputs[1]->getCols() != 1 && vectInputs[1]->getRows() != 1) || sizeOfX < 2)
        {
            throw "gpuInterp: Wrong size for input arguments #2: A vector of size at least 2 expected.";
        }

        if ( vectInputs[1]->getRows() != vectInputs[2]->getRows() ||
                vectInputs[1]->getCols() != vectInputs[2]->getCols() ||
                vectInputs[1]->getRows() != vectInputs[3]->getRows() ||
                vectInputs[1]->getCols() != vectInputs[3]->getCols())
        {
            throw "gpuInterp: Wrong size for input arguments #2 to #4: Same sizes expected.";
        }

        // out mode
        if (Rhs == 5)
        {
            // Get var env
            sciErr = getVarAddressFromPosition(pvApiCtx, 5, &piAddr);
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
                throw "gpuInterp: Wrong type for input argument #5: A String expected.";
            }

            //fisrt call to retrieve dimensions
            sciErr = getMatrixOfString(pvApiCtx, piAddr, &iRows, &iCols, NULL, NULL);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            if (iRows * iCols != 1)
            {
                throw "gpuInterp: Wrong size for input argument #5: A scalar String expected.";
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

            if (strcmp(pstData, "C0") == 0)
            {
                iType = 8;
            }
            else if (strcmp(pstData, "by_zero") == 0)
            {
                iType = 7;
            }
            else if (strcmp(pstData, "natural") == 0)
            {
                iType = 1;
            }
            else if (strcmp(pstData, "periodic") == 0)
            {
                iType = 3;
            }
            else if (strcmp(pstData, "by_nan") == 0)
            {
                iType = 10;
            }
            else if (strcmp(pstData, "linear") == 0)
            {
                iType = 9;
            }
            else
            {
                char str[100];
                sprintf(str, "gpuInterp : Wrong value for input argument #5 : '%s' is a unknow outmode type.", pstData);
                throw str;
            }

            delete pstData;
        }

#ifdef WITH_OPENCL
        if (!useCuda())
        {
            Scierror(999, "gpuInterp: not implemented with OpenCL.\n");
        }
#endif

#ifdef WITH_CUDA
        if (useCuda())
        {
            for (int i = 0; i < 4; i++)
            {
                tabOutputs[i] = new PointerCuda(vectInputs[0]->getRows(), vectInputs[0]->getCols(), false);
            }

            // *** Perform operation. ***
            cudaError_t cudaStat = interp_gpu(  vectInputs[0]->getGpuPtr(),
                                                tabOutputs[0]->getGpuPtr(),
                                                tabOutputs[1]->getGpuPtr(),
                                                tabOutputs[2]->getGpuPtr(),
                                                tabOutputs[3]->getGpuPtr(),
                                                sizeOfXp,
                                                vectInputs[1]->getGpuPtr(),
                                                vectInputs[2]->getGpuPtr(),
                                                vectInputs[3]->getGpuPtr(),
                                                sizeOfX, iType);

            if (cudaStat != cudaSuccess)
            {
                throw cudaStat;
            }
        }
#endif

        // *** Return result in Scilab. ***
        // Keep the result on the Device.
        for (int i = 0; i < Lhs; i++)
        {
            PointerManager::getInstance()->addGpuPointerInManager(tabOutputs[i]);
            createPointer(pvApiCtx, Rhs + i + 1, (void*)tabOutputs[i]);
            if (sciErr.iErr)
            {
                throw sciErr;
            }
            LhsVar(i + 1) = Rhs + i + 1;
        }

        for (int i = Lhs; i < 4; i++)
        {
            delete tabOutputs[i];
        }

        for (int i = 0; i < 4; i++)
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
#ifdef WITH_CUDA
    catch (cudaError_t cudaE)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaE);
    }
#endif

    // input
    for (int i = 0; i < vectInputs.size(); i++)
    {
        if (!isGpuPtr[i])
        {
            delete vectInputs[i];
        }
    }

    // output
    if (tabOutputs[0])
    {
        for (int i = 0; i < 4; i++)
        {
            delete tabOutputs[i];
        }
    }

    return EXIT_FAILURE;
}
/* ========================================================================== */
