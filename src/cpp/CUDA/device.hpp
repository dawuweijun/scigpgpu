/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/

#ifndef DEVICE_CUDA_HPP_
#define DEVICE_CUDA_HPP_


template<> inline
Device<CUDAmode>::Device()
{

}

template<> inline
int Device<CUDAmode>::initDevice(Device_identifier ordinal)
{
    int major       = 0;
    int minor       = 0;
    int gpuOverlap  = 0;

    int canMapHostMemory = 0;
    char deviceName[256];

    if(__check_sanity__<CUDAmode> ( cuDeviceGet ( &dev, ordinal )))
    {
        return -1;
    }

    if(__check_sanity__<CUDAmode>( cuDeviceComputeCapability(&major,&minor,dev)))
    {
        return -1;
    }

    dev_cap = std::pair<int, int> (major, minor);

    if(__check_sanity__<CUDAmode>( cuDeviceTotalMem(&mem, dev)))
    {
        return -1;
    }

    if(__check_sanity__<CUDAmode>( cuDeviceGetAttribute( &gpuOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, dev )))
    {
        return -1;
    }

    support_cce = (gpuOverlap != 0);
    if(__check_sanity__<CUDAmode>( cuDeviceGetAttribute( &canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev )))
    {
        return -1;
    }

    support_plm = (canMapHostMemory != 0);
    if(__check_sanity__<CUDAmode>( cuDeviceGetName(deviceName, 256, dev)))
    {
        return -1;
    }

    name = std::string(deviceName);

    return 0;
}




#endif /* DEVICE_H_ */
