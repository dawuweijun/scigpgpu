/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
* Copyright (C) Scilab Enterprises - 2012 - Cedric Delamarre
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/

#ifndef CONTEXT_CUDA_HPP_
#define CONTEXT_CUDA_HPP_

/*!
 * \cond
 */

#include <vector>
#include <map>

template<> inline Context<CUDAmode>::Context()
{

}

template<> inline
int Context<CUDAmode>::number_of_device()
{
    int deviceCount = 0;
    if(__check_sanity__<CUDAmode> (cuDeviceGetCount(&deviceCount)) == -1)
    {
        return -1;
    }

    return deviceCount;
}

template<> inline
int Context<CUDAmode>::initContext()
{
    int IDeviceCount = 0;
    if(__check_sanity__<CUDAmode> (cuInit(0)) == -1)
    {
        return -1;
    }

    IDeviceCount = number_of_device();
    for (int ordinal = 0; ordinal < IDeviceCount; ordinal++)
    {
        Device<CUDAmode> device = Device<CUDAmode>();
        if(device.initDevice(ordinal) == -1)
        {
            return -1;
        }
        devices_list.push_back(device);
    }

    return IDeviceCount == -1 ? IDeviceCount : 0;
}

template<>
template<bool isGL> inline
void Context<CUDAmode>::set_current_device(const Device<CUDAmode>& device)
{
    if (isGL)
    {
        __check_sanity__<CUDAmode> (cuGLCtxCreate(&cont, 0, device.dev));
    }
    else
    {
        __check_sanity__<CUDAmode> (cuCtxCreate(&cont, 0, device.dev));
    }

    current_device = device;
}

template<> inline
Context<CUDAmode>::~Context()
{
    loadedModule.clear();
    __check_sanity__ <CUDAmode> (cuCtxDestroy(cont));
}
/*!
 * \endcond
 */

#endif /* CONTEXT_HPP_ */
