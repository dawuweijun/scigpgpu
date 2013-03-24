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
#ifndef __CUDA_WRAP_H__
#define __CUDA_WRAP_H__

#include <string>
#include <vector>
#include <exception>
#include <map>
#include <iostream>

#ifdef __APPLE__
    #include <gl.h>
#else
    #include <GL/gl.h>
#endif

#include <cuda.h>
#include <cudaGL.h>


template<>
struct ModeDefinition<CUDA>
{
public:
    typedef CUcontext Context_Handle;
    typedef CUresult Status;
    typedef CUdevice Device_Handle;
    typedef CUmodule Module_Handle;
    typedef CUdeviceptr DevicePtr_Handle;
    typedef CUfunction Function_Handle;
    typedef CUstream Stream;
    typedef CUgraphicsResource Graphics_Handle;
    typedef CUdevice Platform;
    typedef int Device_identifier;
    enum{
        mode=CUDA
    };
};

#define CUDAmode ModeDefinition<CUDA>

#include <CUDA/gpuerror.hpp>
#include <CUDA/device.hpp>
#include <CUDA/queue.hpp>
#include <CUDA/matrix.hpp>
#include <CUDA/kernel.hpp>
#include <CUDA/module.hpp>
#include <CUDA/context.hpp>

#endif /* __CUDA_WRAP_H__ */
/* ========================================================================== */
