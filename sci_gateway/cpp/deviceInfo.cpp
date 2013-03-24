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
#include <stdlib.h>
#include "deviceInfo.h"
#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef WITH_OPENCL
#ifdef __APPLE__
#include <cl.h>
#else
#include <CL/cl.h>
#endif
#endif
#include "sciprint.h"
#include "Scierror.h"
/* ========================================================================== */
#ifdef WITH_CUDA
int cudaDeviceInfo(void)
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
    {
        Scierror(999, "\ncudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n\n");
        return 1;
    }

    sciprint("Starting...\n\n");
    sciprint(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        sciprint("There is no device supporting CUDA\n");
    }

    int dev = 0;
    int driverVersion = 0, runtimeVersion = 0;
    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0)
        {
            // This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            {
                sciprint("There is no device supporting CUDA.\n");
            }
            else if (deviceCount == 1)
            {
                sciprint("There is 1 device supporting CUDA\n");
            }
            else
            {
                sciprint("There are %d devices supporting CUDA\n", deviceCount);
            }
        }
        sciprint("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

#if CUDART_VERSION >= 2020
        // Console log
        cudaDriverGetVersion(&driverVersion);
        sciprint("  CUDA Driver Version:                           %d.%d\n", driverVersion / 1000, driverVersion % 100);
        cudaRuntimeGetVersion(&runtimeVersion);
        sciprint("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion / 1000, runtimeVersion % 100);
#endif
        sciprint("  CUDA Capability Major revision number:         %d\n", deviceProp.major);
        sciprint("  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);

        sciprint("  Total amount of global memory:                 %u bytes\n", deviceProp.totalGlobalMem);
#if CUDART_VERSION >= 2000
        sciprint("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
        //    sciprint("  Number of cores:                               %d\n", nGpuArchCoresPerSM[deviceProp.major] * deviceProp.multiProcessorCount);
#endif
        sciprint("  Total amount of constant memory:               %u bytes\n", deviceProp.totalConstMem);
        sciprint("  Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
        sciprint("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        sciprint("  Warp size:                                     %d\n", deviceProp.warpSize);
        sciprint("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        sciprint("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
                 deviceProp.maxThreadsDim[0],
                 deviceProp.maxThreadsDim[1],
                 deviceProp.maxThreadsDim[2]);
        sciprint("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
                 deviceProp.maxGridSize[0],
                 deviceProp.maxGridSize[1],
                 deviceProp.maxGridSize[2]);
        sciprint("  Maximum memory pitch:                          %u bytes\n", deviceProp.memPitch);
        sciprint("  Texture alignment:                             %u bytes\n", deviceProp.textureAlignment);
        sciprint("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
#if CUDART_VERSION >= 2000
        sciprint("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
#endif
#if CUDART_VERSION >= 2020
        sciprint("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        sciprint("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
        sciprint("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        sciprint("  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
                 "Default (multiple host threads can use this device simultaneously)" :
                 deviceProp.computeMode == cudaComputeModeExclusive ?
                 "Exclusive (only one host thread at a time can use this device)" :
                 deviceProp.computeMode == cudaComputeModeProhibited ?
                 "Prohibited (no host thread can use this device)" :
                 "Unknown");
#endif
#if CUDART_VERSION >= 3000
        sciprint("  Concurrent kernel execution:                   %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
#endif
#if CUDART_VERSION >= 3010
        sciprint("  Device has ECC support enabled:                %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
#endif
    }
    // finish
    sciprint("\n\nPASSED\n");
    return 0;
}
#endif
/* ========================================================================== */
#ifdef WITH_OPENCL
int OpenClDeviceInfo(void)
{
    cl_int err;
    bool gotit = true;
    cl_uint num_platforms = 0;
    cl_platform_id platform = NULL;
    cl_platform_id* platforms = NULL;

    //INFO STR
    char buffer[1024];

    // Get a platform if any
    gotit = (CL_SUCCESS == clGetPlatformIDs(0, NULL, &num_platforms));
    if (! gotit || (num_platforms == 0) )
    {
        sciprint("ERROR : no platform was found\n");
        return 1;
    }
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id*) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, &num_platforms);

    for (unsigned int i = 0; i < num_platforms; i++)
    {
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buffer), &buffer, NULL);
        sciprint("================================================\n");
        sciprint("PLATFORM NAME :\t %s\n", buffer);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(buffer), &buffer, NULL);
        sciprint ("PLATFORM VERSION : %s\n", buffer);

        cl_uint devices_count = 0;

        err = clGetDeviceIDs (platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &devices_count);
        if (err != CL_SUCCESS)
        {
            sciprint("problem with clGetDevicesIDs : ");
            if ( err  == CL_INVALID_PLATFORM)
                sciprint("CL_INVALID_PLATFORM\n");
            if (err == CL_INVALID_DEVICE_TYPE)
                sciprint("CL_INVALID_DEVICE_TYPE\n");
        }
        if (devices_count == 0)
        {
            sciprint ("No devices found supporting OpenCL\n");
            break;
        }
        else
        {
            sciprint ("%d devices found supporting OpenCL\n", devices_count);
            cl_device_id* devices;
            devices = (cl_device_id*) malloc(sizeof(cl_device_id) * devices_count);
            if (clGetDeviceIDs (platforms[i], CL_DEVICE_TYPE_ALL, devices_count, devices, &devices_count) == CL_SUCCESS)
            {
                for (unsigned int j = 0 ; j < devices_count; j++)
                {
                    clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buffer), &buffer, NULL);
                    sciprint(" Device name :%s\n", buffer);
                    cl_uint nb_units;
                    clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &nb_units, NULL);
                    sciprint(" Max Compute Units : %u\n", nb_units);
                    cl_ulong nb_global_mem;
                    clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &nb_global_mem, NULL);
                    sciprint(" Amount of Global Memory : %u bytes\n", nb_global_mem);
                    cl_ulong nb_local_mem;
                    clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &nb_local_mem, NULL);
                    sciprint(" Amount of Local Memory : %u bytes\n", nb_local_mem);

                    size_t group_size;
                    clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &group_size, NULL);
                    sciprint(" Max Work Group Size : %d\n", group_size);

                    cl_uint nb_dim;
                    clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &nb_dim, NULL);
                    sciprint(" Max Work Item Dimensions : %d\n", nb_dim);
                    size_t* work_size = (size_t*) malloc(sizeof(size_t) * nb_dim);
                    clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t*)*nb_dim, work_size, NULL);
                    for (unsigned int k = 0; k < nb_dim; k++)
                    {
                        sciprint(" Max Work Items on dimemsion %d : %d\n", k, work_size[k]);
                    }
                }
            }
            else
            {
                sciprint("Error in clGetDeviceIDs call\n");
                break;
            }
            free (devices);
        }
    }
    sciprint("================================================\n");
    return 0;
}
#endif
/* ========================================================================== */
