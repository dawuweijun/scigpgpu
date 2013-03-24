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
/* ==================================================================== */
#include "useCuda.h"
#include "config_gpu.h"
/* ==================================================================== */
#ifdef WITH_CUDA
    static int iUseCuda = 1;
#else
    static int iUseCuda = 0;
#endif
/* ==================================================================== */
int setUseCuda(int use)
{
    #ifdef WITH_OPENCL
    #ifdef WITH_CUDA
    iUseCuda = (use == 1);
    #endif
    #endif
    return iUseCuda;
}
/* ==================================================================== */
int useCuda(void)
{
    return iUseCuda;
}
/* ==================================================================== */
