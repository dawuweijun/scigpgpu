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
int sci_gpuLoadFunction(char *fname)
{
    int row = 0, col = 0, length = 0;
    int* pstrA = NULL;
    int* pstrB = NULL;
    int* piLen = NULL;
    char** pstData = NULL;
    char* funcname = NULL;
    SciErr sciErr;
    int inputType_A = 0;
    int inputType_B = 0;

    CheckRhs(2, 2);
    CheckLhs(1, 1);

    try
    {
        if (!isGpuInit())
            throw "gpu is not initialised. Please launch gpuInit() before use this function.";

        sciErr = getVarAddressFromPosition(pvApiCtx, 1, &pstrA);
        if (sciErr.iErr) throw sciErr;

        sciErr = getVarType(pvApiCtx, pstrA, &inputType_A);
        if (sciErr.iErr) throw sciErr;

        sciErr = getVarAddressFromPosition(pvApiCtx, 2, &pstrB);
        if (sciErr.iErr) throw sciErr;

        sciErr = getVarType(pvApiCtx, pstrB, &inputType_B);
        if (sciErr.iErr) throw sciErr;

        if (inputType_A != sci_strings)
        {
            throw "gpuLoadFuntion : Bad type for input argument #1: A matrix of String expected.";
        }
        else if (inputType_B != sci_strings)
        {
            throw "gpuLoadFuntion : Bad type for input argument #2: A matrix of String expected.";
        }

        // get first argument

        sciErr = getMatrixOfString(pvApiCtx, pstrA, &row, &col, NULL, NULL);
        if (sciErr.iErr) throw sciErr;

        if (row * col != 2)
            throw "gpuLoadFunction : Bad size for input argument #1: A matrix of size two expected.";

        piLen = (int*)malloc(sizeof(int) * row * col);
        sciErr = getMatrixOfString(pvApiCtx, pstrA, &row, &col, piLen, NULL);
        if (sciErr.iErr) throw sciErr;

        pstData = (char**)malloc(sizeof(char*) * row * col);
        if (pstData == NULL)
            throw "gpuLoadFunction : memory allocation failed.";

        for (int i = 0 ; i < row * col ; i++)
        {
            pstData[i] = (char*)malloc(sizeof(char) * (piLen[i] + 1));
        }

        sciErr = getMatrixOfString(pvApiCtx, pstrA, &row, &col, piLen, pstData);
        if (sciErr.iErr) throw sciErr;

        // get second argument

        sciErr = getMatrixOfString(pvApiCtx, pstrB, &row, &col, &length, NULL);
        if (sciErr.iErr) throw sciErr;

        if (row*col != 1)
            throw "gpuLoadFunction : Bad size for input argument #2: A scalar expected.";

        funcname = (char*)malloc((length + 1) * sizeof(char));
        if (funcname == NULL)
            throw "gpuLoadFunction : memory allocation failed.";


        sciErr = getMatrixOfString(pvApiCtx, pstrB, &row, &col, &length, &funcname);
        if (sciErr.iErr) throw sciErr;

        // Apply

#ifdef WITH_CUDA
        if (useCuda())
        {
            if (pstData[1] == "OpenCL")
                throw "gpuLoadFunction : Please use OpenCL mode to load this function.";

            const  Module<ModeDefinition<CUDA> >* md = getCudaContext()->getModule(pstData[0]);
            if (md == NULL)
                throw "gpuLoadFunction : Load module failed.";

            Kernel<ModeDefinition<CUDA> >* fonc = md->getFunction(std::string(funcname));
            sciErr = createPointer(pvApiCtx, Rhs + 1, (void*)fonc);
        }
#endif

#ifdef WITH_OPENCL
        if (!useCuda())
        {
            if (pstData[1] == "Cuda")
                throw "gpuLoadFunction : Please use Cuda mode to load this function.";

            const  Module<ModeDefinition<OpenCL> >* md = getOpenClContext()->getModule(pstData[0]);
            if (md == NULL)
                throw "gpuLoadFunction : Load module failed.";

            Kernel<ModeDefinition<OpenCL> >* fonc = md->getFunction(std::string(funcname));
            sciErr = createPointer(pvApiCtx , Rhs + 1, (void*)fonc);
        }
#endif

        if (sciErr.iErr) throw sciErr;

        LhsVar(1) = Rhs + 1;
        PutLhsVar();
    }
    catch (const char* str)
    {
        Scierror(999, "%s: %s\n", fname, str);
    }
    catch (SciErr E)
    {
        printError(&E, 0);
    }


    if (piLen != NULL)
    {
        free(piLen);
        piLen = NULL;
    }
    if (pstData != NULL)
    {
        free(pstData);
        pstData = NULL;
    }

    if (funcname != NULL)
    {
        free(funcname);
        funcname = NULL;
    }
    return 0;
}
/* ========================================================================== */
