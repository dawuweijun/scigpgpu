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
#include "gpuPointerManager.hxx"
/* ========================================================================== */
#include "Scierror.h"
#include "api_scilab.h"
/* ========================================================================== */

int sci_gpuExit(char* fname)
{
    CheckRhs(0, 0);
    CheckLhs(0, 1);

    if (isGpuInit())
    {
        deleteGpuContext();
        gpuNotInitialised();
        PointerManager::killInstance();
        PutLhsVar();
    }
    else
    {
        Scierror(999, "%s", "gpu is not initialised. Please launch gpuInit() before use this function.\n");
    }
    return 0;
}
/* ========================================================================== */
