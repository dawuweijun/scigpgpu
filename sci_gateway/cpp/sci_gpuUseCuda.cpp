/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
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
#include "Scierror.h"
#include "config_gpu.h"
#include "useCuda.h"
#include "gw_gpu.h"
#include "checkDevice.h"
#include "gpuContext.hxx"
#include "api_scilab.h"
/* ========================================================================== */
#define BOOL int
#define TRUE 1
#define FALSE 0

static int sci_gpuUseCuda_NO_RHS(char* fname);
static int sci_gpuUseCuda_ONE_RHS(char* fname);
/* ========================================================================== */
int sci_gpuUseCuda(char* fname)
{
    CheckRhs(0, 1);
    CheckLhs(0, 1);

    if (Rhs == 1)
    {
        return sci_gpuUseCuda_ONE_RHS(fname);
    }
    else
    {
        return sci_gpuUseCuda_NO_RHS(fname);
    }
    return 0;
}
/* ========================================================================== */
static int sci_gpuUseCuda_NO_RHS(char* fname)
{
    createScalarBoolean(pvApiCtx, Rhs + 1, (useCuda() == 1));
    LhsVar(1) = Rhs + 1;
    PutLhsVar();

    return 0;
}
/* ========================================================================== */
static int sci_gpuUseCuda_ONE_RHS(char* fname)
{
    SciErr sciErr;
    BOOL bUse = TRUE;
    int iType = 0;
    int *piAddressVarOne = NULL;

    /* get Address of inputs */
    sciErr = getVarAddressFromPosition(pvApiCtx, 1, &piAddressVarOne);
    if(sciErr.iErr)
    {
        printError(&sciErr, 0);
        return 0;
    }

    if (isBooleanType(pvApiCtx, piAddressVarOne))
    {
        if (isScalar(pvApiCtx, piAddressVarOne))
        {
            getScalarBoolean(pvApiCtx, piAddressVarOne, &bUse);
            setUseCuda(bUse);
            sci_gpuUseCuda_NO_RHS(fname);
        }
        else
        {
            Scierror(999,"%s: Wrong size for input argument #%d: A boolean expected.\n", fname, 1);
        }
    }
    else
    {
        Scierror(999,"%s: Wrong type for input argument #%d: A boolean expected.\n", fname, 1);
    }
	
	if (!isGpuInit())
    {
        setGpuContext(0);
        gpuInitialised();	
    }
    return 0; 
}
/* ========================================================================== */
