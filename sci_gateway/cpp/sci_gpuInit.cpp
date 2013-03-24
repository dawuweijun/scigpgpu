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
#include "checkDevice.h"
#include "gpuContext.hxx"
/* ========================================================================== */
#include "Scierror.h"
#include "api_scilab.h"
/* ========================================================================== */
int sci_gpuInit(char* fname)
{
    CheckRhs(0, 0);
    CheckLhs(0, 1);

    if (!isGpuInit())
    {
        if (setGpuContext(0))
        {
            return 1;
        }

        gpuInitialised();
    }
    /* else
       {
           Scierror(999,"%s","gpu is already initialised.\n");
    }
    */
    PutLhsVar();
    return 0;
}
/* ========================================================================== */
