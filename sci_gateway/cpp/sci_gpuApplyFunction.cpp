/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
* Copyright (C) DIGITEO - 2010-2011 - Cedric DELAMARRE
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
#ifdef WITH_CUDA
static int sci_CUDA_getArgs(Kernel<ModeDefinition<CUDA> >* ker,
                            int* lstptr, int argnum, char *fname);
#endif
#ifdef WITH_OPENCL
static int sci_OpenCL_getArgs(Kernel<ModeDefinition<OpenCL> >* ker,
                              int* lstptr, int argnum, char *fname);
#endif
/* ========================================================================== */
int sci_gpuApplyFunction(char *fname)
{
    SciErr sciErr;
    int block_w = 0, block_h = 0, grid_w = 0, grid_h = 0;
    int row = 0, col = 0;
    int argnum = 0;
    int retnumbvalue = 0;
    double* MM = NULL;

    int *ptr = NULL;
    int *lstptr = NULL;
    int *dptr[4];
    int inputType_1 = 0;
    int inputType_2 = 0;

    CheckRhs(6, 6);
    CheckLhs(1, 1);

    try
    {
        void *fptr = NULL;

        if (!isGpuInit())
        {
            throw "gpu is not initialised. Please launch gpuInit() before use this function.";
        }

        sciErr = getVarAddressFromPosition(pvApiCtx, 1, &ptr);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        sciErr = getVarType(pvApiCtx, ptr, &inputType_1);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        sciErr = getVarAddressFromPosition(pvApiCtx, 2, &lstptr);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        sciErr = getVarType(pvApiCtx, lstptr, &inputType_2);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        if (inputType_1 != sci_pointer)
        {
            throw "gpuApplyFuntion : Bad type for input argument #1: A string expected.";
        }
        else if (inputType_2 != sci_list)
        {
            throw "gpuApplyFuntion : Bad type for input argument #2: A list expected.";
        }
        else
        {
            for (int i = 3; i < 7; i++)
            {
                int inputType = 0;
                sciErr = getVarAddressFromPosition(pvApiCtx, i, &dptr[i - 3]);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }

                sciErr = getVarType(pvApiCtx, dptr[i - 3], &inputType);
                if (sciErr.iErr)
                {
                    throw sciErr;
                }

                if (inputType != sci_matrix)
                {
                    char string[64];
                    sprintf(string, "gpuApplyFunction : Bad type for input argument #%d : A matrix expected.", i);
                    throw string;
                }
            }

            sciErr = getPointer(pvApiCtx, ptr, (void**)&fptr);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            sciErr = getListItemNumber(pvApiCtx, lstptr, &argnum);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            sciErr = getMatrixOfDouble(pvApiCtx, dptr[0], &row, &col, &MM);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            if (row * col != 1)
            {
                throw "gpuApplyFunction : Bad size for input argument #3: A scalar expected.";
            }

            block_h = (int)MM[0];

            sciErr = getMatrixOfDouble(pvApiCtx, dptr[1], &row, &col, &MM);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            if (row * col != 1)
            {
                throw "gpuApplyFunction : Bad size for input argument #4: A scalar expected.";
            }

            block_w = (int)MM[0];

            sciErr = getMatrixOfDouble(pvApiCtx, dptr[2], &row, &col, &MM);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            if (row * col != 1)
            {
                throw "gpuApplyFunction : Bad size for input argument #5: A scalar expected.";
            }

            grid_h = (int)MM[0];

            sciErr = getMatrixOfDouble(pvApiCtx, dptr[3], &row, &col, &MM);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            if (row * col != 1)
            {
                throw "gpuApplyFunction : Bad size for input argument #6: A scalar expected.";
            }

            grid_w = (int)MM[0];

#ifdef WITH_CUDA
            if (useCuda())
            {
                Kernel<ModeDefinition<CUDA> >* fptrCuda = (Kernel<ModeDefinition<CUDA> >*)fptr;
                sci_CUDA_getArgs(fptrCuda, lstptr, argnum, fname);
                fptrCuda->launch(getCudaQueue() , block_w, block_h, grid_w, grid_h);
            }
#endif

#ifdef WITH_OPENCL
            if (!useCuda())
            {
                Kernel<ModeDefinition<OpenCL> >* fptrOpenCL = (Kernel<ModeDefinition<OpenCL> >*)fptr;
                sci_OpenCL_getArgs(fptrOpenCL, lstptr, argnum, fname);
                fptrOpenCL->launch(getOpenClQueue(), block_w, block_h, grid_w * block_w, grid_h * block_h);
            }
#endif

            PutLhsVar();
        }
    }
    catch (const char* str)
    {
        Scierror(999, "%s: %s\n", fname, str);
    }
    catch (SciErr E)
    {
        printError(&E, 0);
    }
    return 0;
}
/* ========================================================================== */
#ifdef WITH_CUDA
static int sci_CUDA_getArgs(Kernel<ModeDefinition<CUDA> >* ker,
                            int* lstptr, int argnum,
                            char *fname)
{
    SciErr  sciErr;
    int* ptr_child = NULL;
    int rowsM   = 0, colsM = 0;
    double *MM  = NULL;
    int iType   = 0;
    double  d   = 0;
    int*    n   = NULL;
    void*   dptr    = NULL;
    PointerCuda* gmat = NULL;

    try
    {
        for (int i = 0; i < argnum; ++i)
        {
            sciErr = getListItemAddress(pvApiCtx, lstptr, i + 1, &ptr_child);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            sciErr = getVarType(pvApiCtx, ptr_child, &iType);
            if (sciErr.iErr)
            {
                throw sciErr;
            }
            switch (iType)
            {
                case sci_pointer:
                {
                    sciErr = getPointer(pvApiCtx, ptr_child, (void**)&dptr);
                    if (sciErr.iErr)
                    {
                        throw sciErr;
                    }
                    gmat = (PointerCuda*)dptr;
                    if (gmat->getGpuType() != GpuPointer::CudaType)
                    {
                        throw "Bad pointer type. Make sure that is a Cuda pointer.";
                    }
                    ker->pass_argument(gmat->getGpuPtr());
                    break;
                }
                case sci_matrix:
                {
                    sciErr = getMatrixOfDouble(pvApiCtx, ptr_child, &rowsM, &colsM, &MM);
                    if (sciErr.iErr)
                    {
                        throw sciErr;
                    }
                    d = MM[0];
                    ker->pass_argument<double>(d);
                    break;
                }
                case sci_ints:
                {
                    sciErr = getMatrixOfInteger32(pvApiCtx, ptr_child, &rowsM, &colsM, &n);
                    if (sciErr.iErr)
                    {
                        throw sciErr;
                    }
                    ker->pass_argument<int>(n[0]);
                    break;
                }
                default:
                    break;
            }
        }
    }
    catch (const char* str)
    {
        Scierror(999, "%s: %s\n", fname, str);
    }
    catch (SciErr E)
    {
        printError(&E, 0);
    }
    return 0;
}
#endif /* WITH_CUDA */
/* ========================================================================== */
#ifdef WITH_OPENCL
static int sci_OpenCL_getArgs(Kernel<ModeDefinition<OpenCL> >* ker,
                              int* lstptr, int argnum, char *fname)
{
    SciErr sciErr;
    int*    ptr_child = NULL;
    int     rowsM   = 0, colsM = 0;
    double *MM  = NULL;
    int     iType   = 0;
    double  d   = 0;
    int*    n   = NULL;
    void*   dptr    = NULL;
    PointerOpenCL* gmat = NULL;

    try
    {
        for (int i = 0; i < argnum; ++i)
        {
            sciErr = getListItemAddress(pvApiCtx, lstptr, i + 1, &ptr_child);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            sciErr = getVarType(pvApiCtx, ptr_child, &iType);
            if (sciErr.iErr)
            {
                throw sciErr;
            }
            switch (iType)
            {
                case sci_pointer:
                {
                    sciErr = getPointer(pvApiCtx, ptr_child, (void**)&dptr);
                    if (sciErr.iErr)
                    {
                        throw sciErr;
                    }

                    gmat = (PointerOpenCL*)dptr;
                    if (gmat->getGpuType() != GpuPointer::OpenCLType)
                    {
                        throw "Bad pointer type. Make sure that is a openCL pointer.";
                    }
                    ker->pass_argument(gmat->getGpuPtr());
                    break;
                }
                case sci_matrix:
                {
                    sciErr = getMatrixOfDouble(pvApiCtx, ptr_child, &rowsM, &colsM, &MM);
                    if (sciErr.iErr)
                    {
                        throw sciErr;
                    }

                    d = MM[0];
                    ker->pass_argument<double>(d);
                    break;
                }
                case sci_ints:
                {
                    sciErr = getMatrixOfInteger32(pvApiCtx, ptr_child, &rowsM, &colsM, &n);
                    if (sciErr.iErr)
                    {
                        throw sciErr;
                    }

                    ker->pass_argument<int>(n[0]);
                    break;
                }
                default:
                    break;
            }
        }
    }
    catch (const char* str)
    {
        Scierror(999, "%s: %s\n", fname, str);
    }
    catch (SciErr E)
    {
        printError(&E, 0);
    }
    return 0;
}
#endif /* WITH_OPENCL */
/* ========================================================================== */
