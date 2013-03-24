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

int sci_gpuInsert(char *fname)
{
    CheckRhs(3, 4);
    CheckLhs(1, 1);

    SciErr sciErr;

    int*    piAddr_A    = NULL;
    int     inputType_A = 0;
    int*    piAddr_B    = NULL;
    int     inputType_B = 0;
    int*    piAddr_R    = NULL;
    int     inputType_R = 0;
    int*    piAddr_C    = NULL;
    int     inputType_C = 0;
    int     rows        = 0;
    int     cols        = 0;
    void*   pvPtr       = NULL;
    int     isScalar    = 0;

    GpuPointer* gpuPtrA   = NULL;
    GpuPointer* gpuPtrB   = NULL;
    GpuPointer* gpuPtrR   = NULL;
    GpuPointer* gpuPtrC   = NULL;
    GpuPointer* gpuPtrPos = NULL;

    try
    {
        if (!isGpuInit())
        {
            throw "gpu is not initialised. Please launch gpuInit() before use this function.";
        }

        //--- Get first argument. In what we insert ---
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

        //--- Get second argument. What we insert ---
        sciErr = getVarAddressFromPosition(pvApiCtx, 2, &piAddr_B);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        sciErr = getVarType(pvApiCtx, piAddr_B, &inputType_B);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        // third and forth arguments are positions where we insert
        //--- get second argument ---
        sciErr = getVarAddressFromPosition(pvApiCtx, 3, &piAddr_R);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        sciErr = getVarType(pvApiCtx, piAddr_R, &inputType_R);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        if (nbInputArgument(pvApiCtx) == 4)
        {
            //--- get third argument ---
            sciErr = getVarAddressFromPosition(pvApiCtx, 4, &piAddr_C);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            sciErr = getVarType(pvApiCtx, piAddr_C, &inputType_C);
            if (sciErr.iErr)
            {
                throw sciErr;
            }
        }

        //--- get input data ---
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
                throw "gpuInsert : Bad type for input argument #1: Variables created with GPU functions expected.";
            }

            if (useCuda() && gpuPtrA->getGpuType() != GpuPointer::CudaType)
            {
                throw "gpuInsert : Bad type for input argument #1: A Cuda pointer expected.";
            }

            if (useCuda() == false && gpuPtrA->getGpuType() != GpuPointer::OpenCLType)
            {
                throw "gpuInsert : Bad type for input argument #1: A OpenCL pointer expected.";
            }
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
                    Scierror(999, "gpuInsert: not implemented with OpenCL.\n");
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
                    Scierror(999, "gpuInsert: not implemented with OpenCL.\n");
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
            throw "gpuInsert : Bad type for input argument #1: A GPU or CPU matrix expected.";
        }

        //--- get what insert ---
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
                throw "gpuInsert : Bad type for input argument #2: Variables created with GPU functions expected.";
            }

            if (useCuda() && gpuPtrA->getGpuType() != GpuPointer::CudaType)
            {
                throw "gpuInsert : Bad type for input argument #2: A Cuda pointer expected.";
            }

            if (useCuda() == false && gpuPtrA->getGpuType() != GpuPointer::OpenCLType)
            {
                throw "gpuInsert : Bad type for input argument #2: A OpenCL pointer expected.";
            }
        }
        else if (inputType_B == sci_matrix)
        {
            double* h = NULL;
            if (isVarComplex(pvApiCtx, piAddr_B))
            {
                double* hi = NULL;
                sciErr = getComplexMatrixOfDouble(pvApiCtx, piAddr_B, &rows, &cols, &h, &hi);
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtrB = new PointerCuda(h, hi, rows, cols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    Scierror(999, "gpuInsert: not implemented with OpenCL.\n");
                }
#endif
            }
            else
            {
                sciErr = getMatrixOfDouble(pvApiCtx, piAddr_B, &rows, &cols, &h);
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtrB = new PointerCuda(h, rows, cols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    Scierror(999, "gpuInsert: not implemented with OpenCL.\n");
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
            throw "gpuInsert : Bad type for input argument #2: A GPU or CPU matrix expected.";
        }

        //--- Get vector of positions or rows positions---
        if (inputType_R == sci_pointer)
        {
            sciErr = getPointer(pvApiCtx, piAddr_R, (void**)&pvPtr);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            gpuPtrR = (GpuPointer*)pvPtr;
            if (!PointerManager::getInstance()->findGpuPointerInManager(gpuPtrR))
            {
                throw "gpuInsert : Bad type for input argument #3: Variables created with GPU functions expected.";
            }

            if (useCuda() && gpuPtrA->getGpuType() != GpuPointer::CudaType)
            {
                throw "gpuInsert : Bad type for input argument #3: A Cuda pointer expected.";
            }

            if (useCuda() == false && gpuPtrA->getGpuType() != GpuPointer::OpenCLType)
            {
                throw "gpuInsert : Bad type for input argument #3: A OpenCL pointer expected.";
            }

            if (gpuPtrR->isGpuComplex())
            {
                throw "gpuInsert : Bad type for input argument #3: A real scalar or matrix expected.";
            }
        }
        else if (inputType_R == sci_matrix)
        {
            double* h = NULL;
            if (isVarComplex(pvApiCtx, piAddr_R))
            {
                throw "gpuInsert : Bad type for input argument #3: A real scalar or matrix expected.";
            }
            else
            {
                sciErr = getMatrixOfDouble(pvApiCtx, piAddr_R, &rows, &cols, &h);
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtrR = new PointerCuda(h, rows, cols);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    Scierror(999, "gpuInsert: not implemented with OpenCL.\n");
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
            throw "gpuInsert : Bad type for input argument #3: A GPU or CPU scalar or matrix expected.";
        }

        //--- Get cols positions ---
        if (nbInputArgument(pvApiCtx) == 4)
        {
            if (gpuPtrR->getCols() != 1)
            {
                throw "gpuInsert : Wrong size for input argument #3: Colomn vector expected.";
            }

            if (inputType_C == sci_pointer)
            {
                sciErr = getPointer(pvApiCtx, piAddr_C, (void**)&pvPtr);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }

                gpuPtrC = (GpuPointer*)pvPtr;
                if (!PointerManager::getInstance()->findGpuPointerInManager(gpuPtrC))
                {
                    throw "gpuInsert : Bad type for input argument #4: Variables created with GPU functions expected.";
                }

                if (useCuda() && gpuPtrA->getGpuType() != GpuPointer::CudaType)
                {
                    throw "gpuInsert : Bad type for input argument #4: A Cuda pointer expected.";
                }

                if (useCuda() == false && gpuPtrA->getGpuType() != GpuPointer::OpenCLType)
                {
                    throw "gpuInsert : Bad type for input argument #4: A OpenCL pointer expected.";
                }

                if (gpuPtrC->isGpuComplex())
                {
                    throw "gpuInsert : Bad type for input argument #4: A real scalar or matrix expected.";
                }
            }
            else if (inputType_C == sci_matrix)
            {
                double* h = NULL;
                if (isVarComplex(pvApiCtx, piAddr_R))
                {
                    throw "gpuInsert : Bad type for input argument #4: A real scalar or matrix expected.";
                }
                else
                {
                    sciErr = getMatrixOfDouble(pvApiCtx, piAddr_C, &rows, &cols, &h);
#ifdef WITH_CUDA
                    if (useCuda())
                    {
                        gpuPtrC = new PointerCuda(h, rows, cols);
                    }
#endif
#ifdef WITH_OPENCL
                    if (!useCuda())
                    {
                        Scierror(999, "gpuInsert: not implemented with OpenCL.\n");
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
                throw "gpuInsert : Bad type for input argument #4: A GPU or CPU scalar or matrix expected.";
            }

            if (gpuPtrC->getRows() != 1)
            {
                throw "gpuInsert : Wrong size for input argument #4: Row vector expected.";
            }

#ifdef WITH_OPENCL
            if (!useCuda())
            {
                throw "gpuInsert : not yet implemented with OpenCL.";
            }
#endif

            double dblARows = (double)gpuPtrA->getRows();

            GpuPointer* onesRows    = NULL;
            GpuPointer* onesCols    = NULL;
            GpuPointer* One         = NULL;
            GpuPointer* gpuPtrARows = NULL;

#ifdef WITH_CUDA
            if (useCuda())
            {
                onesRows    = new PointerCuda(gpuPtrR->getRows(), 1, false);
                onesCols    = new PointerCuda(1, gpuPtrC->getCols(), false);
                One         = new PointerCuda(1, 1, false);
                gpuPtrARows = new PointerCuda(&dblARows, 1, 1);
            }
#endif

            onesRows->initMatrix(1);
            onesCols->initMatrix(1);
            One->initMatrix(1);

            GpuPointer* gpuPtrM = *gpuPtrC - *One;
            GpuPointer* gpuPtr1 = *onesRows * *gpuPtrM;
            GpuPointer* gpuPtr2 = *gpuPtr1  * *gpuPtrARows;
            GpuPointer* gpuPtr3 = *gpuPtrR  * *onesCols;
            gpuPtrPos = *gpuPtr2 + *gpuPtr3;

            delete onesRows;
            delete onesCols;
            delete One;
            delete gpuPtr1;
            delete gpuPtr2;
            delete gpuPtr3;
            delete gpuPtrARows;
            delete gpuPtrM;
        }
        else
        {
            gpuPtrPos = gpuPtrR->clone();
            gpuPtrPos->setRows(gpuPtrR->getSize());
            gpuPtrPos->setCols(1);
        }

        if (gpuPtrB->getSize() == 1)
        {
            isScalar = 1;
        }
        else if (gpuPtrB->getSize() != gpuPtrPos->getSize())
        {
            throw "gpuInsert : Invalid index.";
        }

        // perform operation
#ifdef WITH_OPENCL
        if (!useCuda())
        {
            throw "gpuInsert: not implemented with OpenCL.";
        }
#endif
        int iErr = gpuPtrA->insert(gpuPtrB, gpuPtrPos, isScalar);
        if (iErr)
        {
            throw "gpuInsert : Invalid index.";
        }

        if (inputType_A == 1 && gpuPtrA != NULL)
        {
            delete gpuPtrA;
        }

        if (inputType_B == 1 && gpuPtrB != NULL)
        {
            delete gpuPtrB;
        }

        if (inputType_R == 1 && gpuPtrR != NULL)
        {
            delete gpuPtrR;
        }

        if (gpuPtrC != NULL && inputType_C == 1)
        {
            delete gpuPtrC;
        }

        delete gpuPtrPos;

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

    if (inputType_R == 1 && gpuPtrR != NULL)
    {
        delete gpuPtrR;
    }

    if (inputType_C == 1 && gpuPtrC != NULL)
    {
        delete gpuPtrC;
    }

    if (gpuPtrPos)
    {
        delete gpuPtrPos;
    }

    return EXIT_FAILURE;
}
/* ========================================================================== */
