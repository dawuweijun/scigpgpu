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
#include "useCuda.h"
#include "gw_gpu.h"
#include "deviceInfo.h"
#include "api_scilab.h"
/* ========================================================================== */
int sci_gpuDeviceInfo(char *fname)
{
    int iErr = 0;
#ifdef WITH_CUDA
    if (useCuda())
    {
        iErr = cudaDeviceInfo();
    }
#endif

#ifdef WITH_OPENCL
    if (!useCuda())
    {
        iErr = OpenClDeviceInfo();
    }
#endif
    PutLhsVar();
    return iErr;
}
/* ========================================================================== */
