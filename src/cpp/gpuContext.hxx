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
#ifndef __GPUCONTEXT_HXX__
#define __GPUCONTEXT_HXX__

#include "gpu_wrap.h"
#include "config_gpu.h"
#include "dynlib_gpu.h"

GPU_IMPEXP int setGpuContext(int iDevice);

GPU_IMPEXP int deleteGpuContext(void);

#ifdef WITH_CUDA
GPU_IMPEXP Context<ModeDefinition<CUDA> >* getCudaContext(void);
GPU_IMPEXP Queue<ModeDefinition<CUDA> > getCudaQueue(void);
#endif

#ifdef WITH_OPENCL
GPU_IMPEXP Context<ModeDefinition<OpenCL> >* getOpenClContext(void);
GPU_IMPEXP Queue<ModeDefinition<OpenCL> > getOpenClQueue(void);
#endif

#endif /*  __GPUCONTEXT_HXX__ */
/* ========================================================================== */
