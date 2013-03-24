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

#ifndef MATRIX_CUDA_H_
#define MATRIX_CUDA_H_
#include <cuda.h>
template<typename T>
class Matrix<CUDAmode, T> : public MatrixBase<T>
{
	friend class Kernel<CUDAmode> ;
	friend class MatrixFactory<CUDAmode, T> ;
	typedef typename CUDAmode::Status Status;
	typedef typename CUDAmode::DevicePtr_Handle DevicePtr_Handle;
	typedef typename CUDAmode::Context_Handle Context_Handle;
	typedef typename CUDAmode::Stream Stream;
	typedef MatrixBase<T> Base;

protected:
	Stream stream;
	Context_Handle cont;
	DevicePtr_Handle ptr;
/*
private:
    T* cpuMatrix;
*/
public:
	inline
	Matrix(CUDAmode::Context_Handle, CUDAmode::Stream, int n, T* tmp) : Base(n)
	{
		__check_sanity__<CUDAmode>(cuMemAlloc(&ptr, (Base::length) * sizeof(T)));
		if (tmp != NULL)
		{
			__check_sanity__<CUDAmode> (cuMemcpyHtoD(ptr, tmp, (Base::length) * sizeof(T)));
		}
	}

	inline Matrix()
	{

	}

	inline Matrix(const Matrix& other_matrix)
	{
		Base::length = other_matrix.length;
		__check_sanity__<CUDAmode> (cuMemAlloc(&ptr, Base::length * sizeof(T)));
		__check_sanity__<CUDAmode> (cuMemcpyDtoD(ptr, other_matrix.ptr, Base::length * sizeof(T)));
	}

	inline const Matrix<CUDAmode, T>& operator=(const Matrix<CUDAmode, T>& input)
	{
		if (ptr)
		{
			__check_sanity__<CUDAmode> (cuMemFree(ptr));
		}

		Base::length = input.length;
		__check_sanity__<CUDAmode> (cuMemAlloc(&ptr, Base::length * sizeof(T)));
		__check_sanity__<CUDAmode> (cuMemcpyDtoD(ptr, input.ptr, Base::length * sizeof(T)));

		return *this;
	}

	inline typename CUDAmode::DevicePtr_Handle& get_ptr()
	{
		return ptr;
	}

	inline void set_ptr(double* newPtr)
	{
		if (ptr)
			__check_sanity__<CUDAmode> (cuMemFree(ptr));

		ptr = (DevicePtr_Handle&) newPtr;
	}

	inline ~Matrix()
	{
	/*	unsigned int free,total;
		cuMemGetInfo(&free,&total);
		cout<<"Total : "<<total<<endl<<"Free  : "<<free<<endl;
	*/
		if (ptr)
			__check_sanity__<CUDAmode> (cuMemFree(ptr));
	/*
		cuMemGetInfo(&free,&total);
		cout<<"Free  : "<<free<<endl;
	*/
	}
/*
	inline T* to_cpu_ptr() const
	{
		cpuMatrix = new T[Base::length];
		__check_sanity__<CUDAmode> (cuMemcpyDtoH(cpuMatrix, ptr, Base::length
				* sizeof(T)));
		return cpuMatrix;
	}
*/
	inline void to_cpu_ptr(T* mat) const
	{
		__check_sanity__<CUDAmode> (cuMemcpyDtoH(mat, ptr, Base::length * sizeof(T)));
	}

/*    inline void delete_cpu_ptr() const
    {
        delete cpuMatrix;
    }
*/
};

template<typename T>
class GLMatrix<CUDAmode, T> : public MatrixBase<T>
{
	typedef typename CUDAmode::Status Status;
	typedef typename CUDAmode::DevicePtr_Handle DevicePtr_Handle;
	typedef typename CUDAmode::Graphics_Handle Graphics_Handle;
	typedef MatrixBase<T> Base;
protected:
	Graphics_Handle vbo;

public:

	GLMatrix(CUDAmode::Context_Handle, CUDAmode::Stream,GLuint ptr)
	{
		__check_sanity__<CUDAmode> (cuGraphicsGLRegisterBuffer(&vbo, ptr,
				CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE));
	}

	typename CUDAmode::DevicePtr_Handle map_ptr()
	{
		typename CUDAmode::DevicePtr_Handle ptr;
		__check_sanity__<CUDAmode> (cuGraphicsMapResources(1, &vbo, 0));
		size_t tmp;// = Base::length * sizeof(T);
		__check_sanity__<CUDAmode> (cuGraphicsResourceGetMappedPointer(&ptr,
				&tmp, vbo));
		Base::length = tmp / sizeof(T);

		return ptr;
	}

	void unmap()
	{
		__check_sanity__<CUDAmode> (cuGraphicsUnmapResources(1, &vbo, 0));
	}
};

#endif
