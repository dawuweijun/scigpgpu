/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
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
#ifndef __DEVICEINFO_H__
#define __DEVICEINFO_H__

#include "config_gpu.h"

#ifdef WITH_CUDA
int cudaDeviceInfo(void);
#endif

#ifdef WITH_OPENCL
int OpenClDeviceInfo(void);
#endif

#endif /* __DEVICEINFO_H__ */
/* ========================================================================== */
