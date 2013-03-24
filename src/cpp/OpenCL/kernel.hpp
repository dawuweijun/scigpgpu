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

#ifndef KERNEL_OPENCL_HPP_
#define KERNEL_OPENCL_HPP_

#include <memory> // std::shared_ptr

template<>
inline Kernel<OPENCLmode>::Kernel():offset(0)
{

}

template<>
inline Kernel<OPENCLmode>::Kernel(Function_Handle fptr):offset(0), fonc(fptr)
{

}

template<>
template<typename T>
inline void Kernel<OPENCLmode>::pass_argument(std::shared_ptr<Matrix<OPENCLmode,T> > input)
{
	__check_sanity__<OPENCLmode>( clSetKernelArg(fonc,offset,sizeof(cl_mem),static_cast<void*> (&((*input).ptr))) );
    offset++;
}

template<>
template<typename T>
inline void Kernel<OPENCLmode>::pass_argument(std::shared_ptr<GLMatrix<OPENCLmode, T> > input)
{
	cl_mem ptr=(*input).map_ptr();
	__check_sanity__<OPENCLmode>( clSetKernelArg(fonc,offset,sizeof(cl_mem),static_cast<void*> (&ptr)) );
	    offset++;
}

template<>
template<typename T>
inline void Kernel<OPENCLmode>::pass_argument(T f)
{
	T f2(f);
	__check_sanity__<OPENCLmode>( clSetKernelArg(fonc,offset,sizeof(T),static_cast<void*>(&f2)) );
    offset++;
}

template<>
inline void Kernel<OPENCLmode>::launch(Queue<OPENCLmode> queue,int block_w, int block_h, int grid_w, int grid_h)
{
    size_t global_work_size[2];
    size_t local_work_size[2];

    global_work_size[0]=grid_w;
    global_work_size[1]=grid_h;
    local_work_size[0]=block_w;
    local_work_size[1]=block_h;

    __check_sanity__<OPENCLmode>( clEnqueueNDRangeKernel(queue.stream,fonc,2,0,global_work_size,local_work_size,0,0,0) );
    offset=0;
}


#endif /* KERNEL_H_ */
