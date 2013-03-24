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
#include "with_opencl.h"
#include "config_gpu.h"
/* ==================================================================== */
int with_opencl(void)
{
    #ifdef WITH_OPENCL
        return 1;
    #endif
    return 0;
}
/* ==================================================================== */
