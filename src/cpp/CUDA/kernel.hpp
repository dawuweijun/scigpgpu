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

/*!
 * \cond
 */

#ifndef KERNEL_CUDA_HPP_
#define KERNEL_CUDA_HPP_

#define ALIGN_UP(off,alignment) \
  (off) = ( (off) +  ( alignment ) - 1 ) & ~ ( ( alignment ) - 1 );

#include <memory> // std::shared_ptr

template<>
inline
Kernel<CUDAmode>::Kernel():offset(0)
{

}

template<>
inline
Kernel<CUDAmode>::Kernel(Function_Handle fptr):offset(0), fonc(fptr)
{

}

template<>
template<typename T>
inline
void Kernel<CUDAmode>::pass_argument(std::shared_ptr<Matrix<CUDAmode,T> > input)
{
	void* tmpptr = (void*) (size_t) (*input).get_ptr();
	ALIGN_UP(offset,__alignof(tmpptr))
	__check_sanity__<CUDAmode> ( cuParamSetv ( fonc, offset, &tmpptr, sizeof ( tmpptr ) ) );
	offset += sizeof(tmpptr);
}

template<>
template<typename T>
inline
void Kernel<CUDAmode>::pass_argument(std::shared_ptr<GLMatrix<CUDAmode, T> > input)
{
	void* tmpptr = (void*) (size_t) (*input).map_ptr();
	ALIGN_UP(offset,__alignof(tmpptr))
	__check_sanity__<CUDAmode> ( cuParamSetv ( fonc, offset, &tmpptr, sizeof ( tmpptr ) ) );
	offset += sizeof(tmpptr);
}

template<>
template<typename T>
inline
void Kernel<CUDAmode>::pass_argument(T f)
{
	ALIGN_UP( offset,__alignof(f))
	__check_sanity__<CUDAmode>(cuParamSetv(fonc, offset, static_cast<void*> (&f), sizeof(f)));
	offset += sizeof(f);
}

template<>
inline
void Kernel<CUDAmode>::launch(Queue<CUDAmode> queue,int block_w, int block_h, int grid_w, int grid_h)
{
	int xdim=block_w;
	int ydim=block_h;
	int g_w=grid_w;	//ced : g_w = (grid_w/block_w)+1;
	int g_h=grid_h;	//ced : g_h = (grid_h/block_h)+1;
	__check_sanity__<CUDAmode> ( cuParamSetSize ( fonc,offset ) );
	__check_sanity__<CUDAmode> ( cuFuncSetBlockShape ( fonc, xdim, ydim, 1 ) );
	__check_sanity__<CUDAmode> ( cuLaunchGrid ( fonc, g_w, g_h ) );
	offset = 0;
}

/*!
 * \endcond
 */

#endif /* KERNEL_H_ */
