/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
* Copyright (C) DIGITEO - 2011 - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/

/* ========================================================================== */
#include <iostream>
#include <fstream>
#include <string.h>
/* ========================================================================== */
#include "config_gpu.h"
#ifdef WITH_OPENCL
#include "builderOpenCL.h"
#endif
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

int sci_gpuBuild(char* fname)
{
    CheckLhs(1, 1);
    CheckRhs(1, 2);

    SciErr sciErr;
    int* piAddr = NULL;

#ifdef WITH_CUDA
    if (useCuda())
    {
        sciErr = getVarAddressFromPosition(pvApiCtx, 1, &piAddr);
        if (sciErr.iErr)
        {
            throw sciErr;
        }

        if (isStringType(pvApiCtx, piAddr) == false)
        {
            throw "gpuBuild : Bad type for input argument #1: a String expected.";
        }

        OverLoad(1);
        return 0;
    }
#endif

#ifdef WITH_OPENCL
    if (!useCuda())
    {
        int inputType;
        int rows;
        int cols;
        int length  = 0;
        char* fileName = NULL;
        std::string  output[2];
        Builder builder_context;
        char* ppstr[2];

        try
        {
            if (!isGpuInit())
            {
                throw "gpu is not initialised. Please launch gpuInit() before use this function.";
            }

            sciErr = getVarAddressFromPosition(pvApiCtx, 1, &piAddr);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            sciErr = getVarType(pvApiCtx, piAddr, &inputType);
            if (sciErr.iErr)
            {
                throw sciErr;
            }
            if (inputType != sci_strings)
            {
                throw "gpuBuild : Bad type for input argument #1: a String expected.";
            }

            sciErr = getMatrixOfString(pvApiCtx, piAddr, &rows, &cols, &length, NULL);
            if (sciErr.iErr)
            {
                throw sciErr;
            }
            if (rows * cols != 1)
            {
                throw "gpuBuild : Bad size for input argument #1: a scalar expected.";
            }

            fileName = (char*)malloc((length + 1) * sizeof(char));
            sciErr = getMatrixOfString(pvApiCtx, piAddr, &rows, &cols, &length, &fileName);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            builder_context.set_current_device<false>(builder_context.get_devices_list().at(0));
            builder_context.build(fileName, "-Werror");

            output[0] = std::string(fileName) + std::string(".cl.out");

            ppstr[0] = (char*)output[0].c_str();
            ppstr[1] = "OpenCL";
            sciErr = createMatrixOfString(pvApiCtx, Rhs + 1, 2, 1, ppstr);
            if (sciErr.iErr)
            {
                throw sciErr;
            }

            LhsVar(1) = Rhs + 1;
            PutLhsVar();

            if (fileName != NULL)
            {
                free(fileName);
                fileName = NULL;
            }

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
        catch (GpuError Ex)
        {
            Scierror(999, "%s\n", Ex.what());
        }

        if (fileName != NULL)
        {
            free(fileName);
        }
        {
            return EXIT_FAILURE;
        }
    }
#endif
}
/* ========================================================================== */
