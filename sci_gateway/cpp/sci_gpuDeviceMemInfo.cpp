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
int sci_gpuDeviceMemInfo(char *fname)
{
    #ifdef WITH_CUDA
    if(isGpuInit())
    {
        if (useCuda())
        {
            size_t free = 0, total = 0;
            cuMemGetInfo(&free,&total);
            double freeMem = (double)free;

            createScalarDouble(pvApiCtx, Rhs + 1, freeMem);
        }
        else
        {
            double zero = 0.;
            createScalarDouble(pvApiCtx, Rhs + 1, zero);
            sciprint("not implemented with OpenCL.\n");
        }

        LhsVar(1) = Rhs + 1;
        PutLhsVar();
    }
    else
    {
        Scierror(999,"%s","gpu is not initialised. Please launch gpuInit() before use this function.\n");
    }

    #else
        sciprint("not implemented with OpenCL.\n");
    #endif
    return 0;
}
/* ========================================================================== */
