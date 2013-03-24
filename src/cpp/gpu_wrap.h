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
#ifndef __CUSTOM_WRAP_H__
#define __CUSTOM_WRAP_H__

#include "config_gpu.h"

enum {
    #ifdef WITH_CUDA
        CUDA,
    #endif
    #ifdef WITH_OPENCL
        OpenCL
    #endif
};

class GpuError;

template<typename ModeDefinition>
class Device;

template<typename ModeDefinition>
class Context;

template<typename ModeDefinition,typename T>
class Matrix;

template<typename ModeDefinition,typename T>
class GLMatrix;

template<typename ModeDefinition>
class Kernel;

template<typename ModeDefinition>
class Queue;

template<typename ModeDefinition>
class Module;

template<class ModeDefinition,typename T>
class MatrixFactory;

template<int mode>
class ModeDefinition;

#include <Custom/gpuerror.hpp>
#include <Custom/device.hpp>
#include <Custom/module.hpp>
#include <Custom/queue.hpp>
#include <Custom/matrix.hpp>
#include <Custom/kernel.hpp>
#include <Custom/context.hpp>

template<typename ModeDefinition>
inline int __check_sanity__(typename ModeDefinition::Status id)
{
    return GpuError::treat_error<ModeDefinition>(id);
}

#ifdef WITH_CUDA
#include "cuda_wrap.h"
#endif

#ifdef WITH_OPENCL
#include "opencl_wrap.h"
#endif

#endif /* __CUSTOM_WRAP_H__ */
/* ========================================================================== */
