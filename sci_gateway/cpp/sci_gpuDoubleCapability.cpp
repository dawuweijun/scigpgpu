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
#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif
#include "gpu_wrap.h"
#include "useCuda.h"
#include "gw_gpu.h"
#include "checkDevice.h"
/* ========================================================================== */
#include "api_scilab.h"
#include "Scierror.h"
#include "sciprint.h"
/* ========================================================================== */
int sci_gpuDoubleCapability(char* fname)
{
    CheckRhs(0,0);
    CheckLhs(1,1);

    int iDoubleCapable = 0;

    #ifdef WITH_CUDA
    if (useCuda())
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,0);
        if(prop.major == 1 && prop.minor > 2)
        {
            iDoubleCapable = 1;
        }
        else if(prop.major > 1)
        {
            iDoubleCapable = 1;
        }
    }
    else
    {
        sciprint("not implemented with OpenCL.\n");
        iDoubleCapable = 0;
    }
    #endif

    #ifdef WITH_OPENCL
    if (!useCuda())
    {
        sciprint("not implemented with OpenCL.\n");
        iDoubleCapable = 0;
    }
    #endif

    createScalarBoolean(pvApiCtx, Rhs + 1, (iDoubleCapable == 1));
    LhsVar(1) = Rhs + 1;
    PutLhsVar();

    return 0;
}
/* ========================================================================== */
