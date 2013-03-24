/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2012 - Cedric Delamarre
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
/* ========================================================================== */
int sci_gpuLoadDynLib(char* fname)
{
    int row         = 0;
    int col         = 0;
    int* piLen      = NULL;
    char** pstData  = NULL;
    int* pstrA      = NULL;
    int inputType_A = 0;
    int  i          = 0;

    CheckRhs(1, 1);
    CheckLhs(0, 1);

    // get lib
    getVarAddressFromPosition(pvApiCtx,1,&pstrA);
    getVarType(pvApiCtx, pstrA, &inputType_A);
    getMatrixOfString(pvApiCtx, pstrA, &row, &col, NULL, NULL);
    piLen = (int*)malloc(sizeof(int) * row * col);
    getMatrixOfString(pvApiCtx, pstrA, &row, &col, piLen, NULL);
    pstData = (char**)malloc(sizeof(char*) * row * col);
    for(i = 0 ; i < row * col ; i++)
    {
        pstData[i] = (char*)malloc(sizeof(char) * (piLen[i] + 1));
    }
    getMatrixOfString(pvApiCtx, pstrA, &row, &col, piLen, pstData);

    // open lib
    printf("%s\n", *pstData);
    Sci_dlopen(*pstData);

    LhsVar(1) = Rhs + 1;
    PutLhsVar();

    return 0;
}
/* ========================================================================== */
