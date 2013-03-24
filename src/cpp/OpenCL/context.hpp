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

#ifndef CONTEXT_OPENGL_HPP_
#define CONTEXT_OPENGL_HPP_

/*!
 * \cond
 */

#include <vector>
#include <map>

template<> inline Context<OPENCLmode>::Context()
{

}

template<> inline
int Context<OPENCLmode>::number_of_device()
{
    return -1;
}

template<> inline
int Context<OPENCLmode>::initContext()
{
    cl_uint numplatforms;
    cl_uint devicescount = 0;
    __check_sanity__<OPENCLmode> (clGetPlatformIDs(0, NULL, &numplatforms));
    platforms = new cl_platform_id[numplatforms];
    __check_sanity__<OPENCLmode> (clGetPlatformIDs(numplatforms, platforms,
    NULL));
    Platform platform = platforms[0];


    __check_sanity__<OPENCLmode> (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
    0, 0, &devicescount));
    cl_device_id* devices = new cl_device_id[devicescount];
    __check_sanity__<OPENCLmode> (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
    devicescount, devices, 0));
    for (int k = 0; k < (int)devicescount; ++k)
    {
        Device<OPENCLmode> device = Device<OPENCLmode>();
        device.initDevice(devices[k]);
        devices_list.push_back(device);
    }
    delete[] devices;
	return 0;
}

template<>
template<bool isGL> inline
void Context<OPENCLmode>::set_current_device(const Device<OPENCLmode>& device)
{
    current_device = device;
    cl_int ciErrNum = CL_SUCCESS;
    cont = clCreateContext(0, 1, &(current_device.dev), 0, 0, &ciErrNum);
    __check_sanity__<OPENCLmode> (ciErrNum);
}

template<> inline
Context<OPENCLmode>::~Context()
{
    delete [] platforms;
}

/*!
 * \endcond
 */

#endif /* CONTEXT_HPP_ */
