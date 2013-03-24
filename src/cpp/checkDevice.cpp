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
#include "config_gpu.h"
#include "checkDevice.h"
#include "useCuda.h"
/* ========================================================================== */
static bool cudaInitState = false;
static bool openclInitState = false;
#ifdef WITH_CUDA
static void cudaInitialised(void);
static void cudaNotInitialised(void);
#endif

#ifdef WITH_OPENCL
static void openclInitialised(void);
static void openclNotInitialised(void);
#endif
/* ========================================================================== */
#ifdef WITH_CUDA
static bool cudaIsInit(void)
{
    return cudaInitState;
}
/* ========================================================================== */
static void cudaInitialised(void)
{
    cudaInitState = true;
}
/* ========================================================================== */
static void cudaNotInitialised(void)
{
    cudaInitState = false;
}
#endif
/* ========================================================================== */
#ifdef WITH_OPENCL
static bool openclIsInit(void)
{
    return openclInitState;
}
/* ========================================================================== */
static void openclInitialised(void)
{
    openclInitState = true;
}
/* ========================================================================== */
static void openclNotInitialised(void)
{
    openclInitState = false;
}
#endif
/* ========================================================================== */
int isGpuInit(void)
{
    int isGpuInit = 0;
    #ifdef WITH_CUDA
    if (useCuda())
    {
        isGpuInit = cudaIsInit();
    }
    #endif

    #ifdef WITH_OPENCL
    if (!useCuda())
    {
        isGpuInit = openclIsInit();
    }
    #endif
    return isGpuInit;
}
/* ========================================================================== */
int gpuInitialised(void)
{
    #ifdef WITH_CUDA
    if (useCuda())
    {
        cudaInitialised();
    }
    #endif

    #ifdef WITH_OPENCL
    if (!useCuda())
    {
        openclInitialised();
    }
    #endif
    return isGpuInit();
}
/* ========================================================================== */
int gpuNotInitialised(void)
{
    #ifdef WITH_CUDA
    if (useCuda())
    {
        cudaNotInitialised();
    }
    #endif

    #ifdef WITH_OPENCL
    if (!useCuda())
    {
        openclNotInitialised();
    }
    #endif
    return isGpuInit();
}
/* ========================================================================== */
