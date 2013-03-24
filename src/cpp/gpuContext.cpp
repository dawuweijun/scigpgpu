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
#include "gpuContext.hxx"
#include "checkDevice.h"
#include "useCuda.h"
/* ========================================================================== */
#ifdef WITH_CUDA
static Context<ModeDefinition<CUDA> >* CUDA_context;
static Queue<ModeDefinition<CUDA> > CUDA_queue;
#endif
/* ========================================================================== */
#ifdef WITH_OPENCL
static Context<ModeDefinition<OpenCL> >* OpenCL_context;
static Queue<ModeDefinition<OpenCL> > OpenCL_queue;
#endif
/* ========================================================================== */
int setGpuContext(int iDevice)
{
    int iErr = 0;
#ifdef WITH_CUDA
    if (useCuda())
    {
        CUDA_context = new Context<ModeDefinition<CUDA> >();
        if (CUDA_context->initContext() == -1)
        {
            return 1;
        }

        const std::vector<Device<ModeDefinition<CUDA> > >& devs = CUDA_context->get_devices_list();
        CUDA_context->set_current_device<false>(devs[iDevice]);
        std::pair<int, int> dc = devs[iDevice].device_capability();
        CUDA_queue = CUDA_context->genQueue();
        cublasInit();
    }
#endif

#ifdef WITH_OPENCL
    if (!useCuda())
    {
        OpenCL_context = new Context<ModeDefinition<OpenCL> >();
        if (OpenCL_context->initContext() == -1)
        {
            return 1;
        }
        const std::vector<Device<ModeDefinition<OpenCL> > >& devs = OpenCL_context->get_devices_list();
        OpenCL_context->set_current_device<false>(devs[iDevice]);
        std::pair<int, int> dc = devs[iDevice].device_capability();
        OpenCL_queue = OpenCL_context->genQueue();
    }
#endif
    return 0;
}
/* ========================================================================== */
#ifdef WITH_CUDA
Context<ModeDefinition<CUDA> >* getCudaContext(void)
{
    return CUDA_context;
}
/* ========================================================================== */
Queue<ModeDefinition<CUDA> > getCudaQueue(void)
{
    return CUDA_queue;
}
#endif
/* ========================================================================== */
#ifdef WITH_OPENCL
Context<ModeDefinition<OpenCL> >* getOpenClContext(void)
{
    return OpenCL_context;
}
/* ========================================================================== */
Queue<ModeDefinition<OpenCL> > getOpenClQueue(void)
{
    return OpenCL_queue;
}
#endif
/* ========================================================================== */
int deleteGpuContext(void)
{
#ifdef WITH_CUDA
    if (useCuda())
    {
        // Shutdown cublas
        cublasShutdown();
        delete CUDA_context;
        CUDA_context = NULL;
    }
#endif

#ifdef WITH_OPENCL
    if (!useCuda())
    {
        delete OpenCL_context;
        OpenCL_context = NULL;
    }
#endif
    return 0;
}
/* ========================================================================== */
