/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
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
#ifndef __OPENCL_WRAP_H__
#define __OPENCL_WRAP_H__

#include <string>
#include <vector>
#include <exception>
#include <map>
#include <iostream>

#ifdef __APPLE__
    #include <cl.h>
    #include <cl_gl.h>
#else
    #include <CL/cl.h>
    #include <CL/cl_gl.h>
#endif

template<>
struct ModeDefinition<OpenCL>
{
public:
    typedef cl_context Context_Handle;
    typedef cl_int Status;
    typedef cl_program Module_Handle;
    typedef cl_device_id Device_Handle;
    typedef cl_mem DevicePtr_Handle;
    typedef cl_kernel Function_Handle;
    typedef cl_command_queue Stream;
    typedef cl_mem Graphics_Handle;
    typedef cl_platform_id Platform;
    typedef cl_device_id  Device_identifier;
    enum{
        mode=OpenCL
    };
};

#define OPENCLmode ModeDefinition<OpenCL>

#include <OpenCL/gpuerror.hpp>
#include <OpenCL/device.hpp>
#include <OpenCL/queue.hpp>
#include <OpenCL/matrix.hpp>
#include <OpenCL/kernel.hpp>
#include <OpenCL/module.hpp>
#include <OpenCL/context.hpp>

#endif /* __OPENCL_WRAP_H__ */
/* ========================================================================== */
