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

#ifndef DEVICE_OPENCL_HPP_
#define DEVICE_OPENCL_HPP_


template<> inline
int Device<OPENCLmode>::initDevice(Device_identifier devid)
{
    char tmp_name[1000];
    dev = devid;
    clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(tmp_name),&tmp_name,NULL);
    name=std::string(tmp_name);
    return 0;
}

template<>
inline Device<OPENCLmode>::Device()
{

}



#endif /* DEVICE_H_ */
