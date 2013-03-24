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
#include "cudaRealImgToComplex.hxx"
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

int sci_gpuComplex(char *fname)
{
    CheckRhs(2, 2);
    CheckLhs(1, 1);

    SciErr sciErr;

    int* piAddr_A       = NULL;
    int* piAddr_B       = NULL;

    void* pvPtrA        = NULL;
    void* pvPtrB        = NULL;

    int inputType_A;
    int inputType_B;

    int iRowsA          = 0;
    int iColsA          = 0;
    int iRowsB          = 0;
    int iColsB          = 0;

    double* ha          = NULL;
    double* hb          = NULL;

    bool AIsCpuScalar  = false;
    bool BIsCpuScalar  = false;
    bool freePtrReal   = false;
    bool freePtrImag   = false;

    GpuPointer* gpuPtrA    = NULL;
    GpuPointer* gpuPtrB    = NULL;
    GpuPointer* gpuPtrC    = NULL;
    GpuPointer* gpuPtrReal = NULL;
    GpuPointer* gpuPtrImag = NULL;

    // Get arguments.
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

    try
    {
        if (!isGpuInit())
        {
            throw "gpu is not initialised. Please launch gpuInit() before use this function.";
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
                throw "gpuComplex : Bad type for input argument #1: Variables created with GPU functions expected.";
            }

            if (useCuda() && gpuPtrA->getGpuType() != GpuPointer::CudaType)
            {
                throw "gpuComplex : Bad type for input argument #1: A Cuda pointer expected.";
            }

            if (useCuda() == false && gpuPtrA->getGpuType() != GpuPointer::OpenCLType)
            {
                throw "gpuComplex : Bad type for input argument #1: A OpenCL pointer expected.";
            }

            if (gpuPtrA->isGpuComplex())
            {
                throw "gpuComplex : Bad type for input argument #1: A real expected.";
            }

            iRowsA = gpuPtrA->getRows();
            iColsA = gpuPtrA->getCols();
        }
        else if (inputType_A == sci_matrix)
        {
            if (isVarComplex(pvApiCtx, piAddr_A))
            {
                throw "gpuComplex : Bad type for input argument #1: A real expected.";
            }
            else
            {
                sciErr = getMatrixOfDouble(pvApiCtx, piAddr_A, &iRowsA, &iColsA, &ha);
                if (iRowsA*iColsA == 1)
                {
                    AIsCpuScalar = true;
                }
                else
                {
#ifdef WITH_CUDA
                    if (useCuda())
                    {
                        gpuPtrA = new PointerCuda(ha, iRowsA, iColsA);
                    }
#endif
#ifdef WITH_OPENCL
                    if (!useCuda())
                    {
                        Scierror(999, "gpuComplex: not implemented with OpenCL.\n");
                    }
#endif
                }
            }

            if (sciErr.iErr)
            {
                throw sciErr;
            }
        }
        else
        {
            throw "gpuComplex : Bad type for input argument #1: A GPU or CPU matrix expected.";
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
                throw "gpuComplex : Bad type for input argument #2: Variables created with GPU functions expected.";
            }

            if (useCuda() && gpuPtrB->getGpuType() != GpuPointer::CudaType)
            {
                throw "gpuComplex : Bad type for input argument #2: A Cuda pointer expected.";
            }

            if (useCuda() == false && gpuPtrB->getGpuType() != GpuPointer::OpenCLType)
            {
                throw "gpuComplex : Bad type for input argument #2: A OpenCL pointer expected.";
            }

            if (gpuPtrB->isGpuComplex())
            {
                throw "gpuComplex : Bad type for input argument #2: A real expected.";
            }

            if (AIsCpuScalar && gpuPtrB->getSize() == 1)
            {
#ifdef WITH_CUDA
                if (useCuda())
                {
                    gpuPtrB = new PointerCuda(ha, iRowsA, iColsA);
                }
#endif
#ifdef WITH_OPENCL
                if (!useCuda())
                {
                    Scierror(999, "gpuComplex: not implemented with OpenCL.\n");
                }
#endif
                AIsCpuScalar = false;
            }

            iRowsB = gpuPtrB->getRows();
            iColsB = gpuPtrB->getCols();
        }
        else if (inputType_B == sci_matrix)
        {
            if (isVarComplex(pvApiCtx, piAddr_B))
            {
                throw "gpuComplex : Bad type for input argument #2: A real expected.";
            }
            else
            {
                sciErr = getMatrixOfDouble(pvApiCtx, piAddr_B, &iRowsB, &iColsB, &hb);
                if (iRowsB*iColsB == 1)
                {
                    if (AIsCpuScalar)
                    {
#ifdef WITH_CUDA
                        if (useCuda())
                        {
                            gpuPtrA = new PointerCuda(ha, iRowsA, iColsA);
                            gpuPtrB = new PointerCuda(hb, iRowsB, iColsB);
                        }
#endif
#ifdef WITH_OPENCL
                        if (!useCuda())
                        {
                            Scierror(999, "gpuComplex: not implemented with OpenCL.\n");
                        }
#endif
                        AIsCpuScalar = false;
                    }
                    else
                    {
                        BIsCpuScalar = true;
                    }
                }
                else
                {
#ifdef WITH_CUDA
                    if (useCuda())
                    {
                        gpuPtrB = new PointerCuda(hb, iRowsB, iColsB);
                    }
#endif
#ifdef WITH_OPENCL
                    if (!useCuda())
                    {
                        Scierror(999, "gpuComplex: not implemented with OpenCL.\n");
                    }
#endif
                }
            }

            if (sciErr.iErr)
            {
                throw sciErr;
            }
        }
        else
        {
            throw "gpuComplex : Bad type for input argument #2: A GPU or CPU matrix expected.";
        }

        if (iRowsA * iColsA == 0 || iRowsB * iColsB == 0)
        {
            throw "gpuComplex : Bad size for inputs arguments: Non empty matrix expected.";
        }

        if ((iRowsA * iColsA > 1 && iRowsB * iColsB > 1) && (iColsA != iColsB || iRowsA != iRowsB))
        {
            throw "gpuComplex : Bad size for inputs arguments: Same size expected.";
        }

        if ((AIsCpuScalar || iRowsA*iColsA == 1) && iRowsB * iColsB > 1)
        {
#ifdef WITH_CUDA
            if (useCuda())
            {
                gpuPtrReal = new PointerCuda(iRowsB, iColsB, false);
            }
#endif
#ifdef WITH_OPENCL
            if (!useCuda())
            {
                Scierror(999, "gpuComplex: not implemented with OpenCL.\n");
            }
#endif

            if (AIsCpuScalar)
            {
                gpuPtrReal->initMatrix(*ha);
            }
            else
            {
                double s;
                gpuPtrA->getData(&s);
                gpuPtrReal->initMatrix(s);
            }

            gpuPtrImag = gpuPtrB;
            freePtrReal = true;
        }
        else if (iRowsA*iColsA > 1 && (BIsCpuScalar || iRowsB*iColsB == 1))
        {
#ifdef WITH_CUDA
            if (useCuda())
            {
                gpuPtrImag = new PointerCuda(iRowsA, iColsA, false);
            }
#endif
#ifdef WITH_OPENCL
            if (!useCuda())
            {
                Scierror(999, "gpuComplex: not implemented with OpenCL.\n");
            }
#endif
            if (BIsCpuScalar)
            {
                gpuPtrImag->initMatrix(*hb);
            }
            else
            {
                double s;
                gpuPtrB->getData(&s);
                gpuPtrImag->initMatrix(s);
            }

            gpuPtrReal = gpuPtrA;
            freePtrImag = true;
        }
        else
        {
            gpuPtrReal = gpuPtrA;
            gpuPtrImag = gpuPtrB;
        }

        //performe operation.
#ifdef WITH_CUDA
        if (useCuda())
        {
            gpuPtrC = cudaRealImgToComplex(dynamic_cast<PointerCuda*>(gpuPtrReal), dynamic_cast<PointerCuda*>(gpuPtrImag));
        }
#endif
#ifdef WITH_OPENCL
        if (!useCuda())
        {
            Scierror(999, "gpuComplex: not implemented with OpenCL.\n");
        }
#endif

        // return result.
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

        if (freePtrReal)
        {
            delete gpuPtrReal;
        }

        if (freePtrImag)
        {
            delete gpuPtrImag;
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

    return EXIT_FAILURE;
}
/* ========================================================================== */
