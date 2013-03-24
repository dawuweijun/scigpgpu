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

#ifndef MATRIX_OPENCL_H_
#define MATRIX_OPENCL_H_

template<typename T>
class Matrix<OPENCLmode, T> : public MatrixBase<T>
{
	friend class Kernel<OPENCLmode> ;
	friend class MatrixFactory<OPENCLmode, T> ;
	typedef typename OPENCLmode::Status Status;
	typedef typename OPENCLmode::DevicePtr_Handle DevicePtr_Handle;
	typedef typename OPENCLmode::Context_Handle Context_Handle;
	typedef typename OPENCLmode::Stream Stream;
	typedef MatrixBase<T> Base;
protected:
	Stream stream;
	Context_Handle cont;
	DevicePtr_Handle ptr;

public:
	inline Matrix(Context_Handle c, Stream s, int n, T* tmp) :
		Base(n), cont(c), stream(s)
	{
		cl_int err_code;
		if (tmp != NULL)
			ptr = clCreateBuffer(cont,
					CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Base::length
							* sizeof(T), tmp, &err_code);
		else
			ptr = clCreateBuffer(cont, CL_MEM_READ_WRITE, Base::length
					* sizeof(T), NULL, &err_code);
		__check_sanity__<OPENCLmode> (err_code);
	}

	inline Matrix(const Matrix& other_matrix)
	{
		Base::length = other_matrix.length;
		__check_sanity__<OPENCLmode>(clEnqueueCopyBuffer(stream, other_matrix.ptr, ptr, 0,
				0, Base::length * sizeof(T), 0, NULL, NULL));
	}

	inline const Matrix<OPENCLmode, T>& operator=(const Matrix<OPENCLmode, T>& input)
	{
		Base::length = input.length;
		__check_sanity__(clEnqueueCopyBuffer(stream, input.ptr, ptr, 0, 0,
				Base::length * sizeof(T), 0, NULL, NULL));
		return *this;
	}

	inline DevicePtr_Handle& get_ptr()
	{
		return ptr;
	}

	inline ~Matrix()
	{
		if (ptr)
		{
			__check_sanity__<OPENCLmode> (clReleaseMemObject(ptr));
		}
	}

	inline void set_ptr(double* newPtr)
	{
		if (ptr)
			__check_sanity__<OPENCLmode> (clReleaseMemObject(ptr));

		ptr = (DevicePtr_Handle&) newPtr;
	}
/*
	inline T* to_cpu_ptr() const
	{
        cpuMatrix = new T[Base::length];
		__check_sanity__<OPENCLmode> (clEnqueueReadBuffer(stream, ptr, CL_TRUE,
				0, Base::length * sizeof(T), static_cast<void*> (cpuMatrix), 0, 0, 0));
		return cpuMatrix;
	}
*/
	inline void to_cpu_ptr(T* mat) const
	{
        __check_sanity__<OPENCLmode> (clEnqueueReadBuffer(stream, ptr, CL_TRUE,
				0, Base::length * sizeof(T), static_cast<void*> (mat), 0, 0, 0));
	}
};

template<typename T>
class GLMatrix<OPENCLmode,T> : MatrixBase<T>
{
	typedef OPENCLmode::Context_Handle Context_Handle;
	typedef OPENCLmode::Stream Stream;
	typedef OPENCLmode::Status Status;
	typedef OPENCLmode::DevicePtr_Handle DevicePtr_Handle;
	typedef OPENCLmode::Graphics_Handle Graphics_Handle;
	typedef MatrixBase<T> Base;

protected:
	Graphics_Handle vbo;
	Stream stream;
public:
	inline
	GLMatrix(Context_Handle c, Stream s,GLuint ptr):stream(s)
	{
		cl_int errcode;
		vbo = clCreateFromGLBuffer(c,CL_MEM_READ_WRITE,ptr,&errcode);
		__check_sanity__<OPENCLmode>(errcode);
	}
	inline
	DevicePtr_Handle map_ptr()
	{
		__check_sanity__<OPENCLmode>(clEnqueueAcquireGLObjects(stream,1,&vbo,0,0,0));
		return vbo;
	}

	inline
	void unmap()
	{
		__check_sanity__<OPENCLmode>(clEnqueueReleaseGLObjects(stream,1,&vbo,0,0,0));
		__check_sanity__<OPENCLmode>(clFinish(stream));
	}
};

#endif
