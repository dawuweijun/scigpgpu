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

#ifndef QUEUE_OPENCL_HPP_
#define QUEUE_OPENCL_HPP_

template<>
inline Queue<OPENCLmode>::Queue()
{

}



template<>
inline Queue<OPENCLmode>::Queue(Context_Handle c,Device_Handle d):cont(c),dev(d)
{
	cl_int ciErrNum = CL_SUCCESS;
	stream=clCreateCommandQueue(cont,dev,0,&ciErrNum);
	__check_sanity__<OPENCLmode>(ciErrNum);
}

#endif /* QUEUE_H_ */
