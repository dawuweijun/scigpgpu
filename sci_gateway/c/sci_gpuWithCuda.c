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
#include "api_scilab.h"
#include "Scierror.h"
#include "with_cuda.h"
/* ========================================================================== */
int sci_gpuWithCuda(char* fname)
{
    CheckRhs(0, 0);
    CheckLhs(0, 1);
    
    createScalarBoolean(pvApiCtx, Rhs + 1, (with_cuda() == 1));
    LhsVar(1) = Rhs + 1;
    PutLhsVar();
    
    return 0;
}
/* ========================================================================== */
