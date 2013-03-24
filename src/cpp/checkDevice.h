/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/
/* ========================================================================== */
#ifndef CHECKDEVICE_H
#define CHECKDEVICE_H
/* ========================================================================== */
#include <cstdlib>
#include <cstdio>
/* ========================================================================== */
#include "config_gpu.h"
/* ========================================================================== */
#ifdef __cplusplus
extern "C"{
#endif

    int isGpuInit(void);

    int gpuInitialised(void);

    int gpuNotInitialised(void);

#ifdef __cplusplus
}
#endif /* extern "C" */

#endif // CHECKDEVICE_H
/* ========================================================================== */
