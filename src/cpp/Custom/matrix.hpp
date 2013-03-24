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

#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#ifdef _MSC_VER
    #include <windows.h>
#endif

#ifdef __APPLE__
    #include <gl.h>
#else
    #include <GL/gl.h>
#endif

template<typename T>
class MatrixBase
{
protected:
    int length;

public:
    inline
    MatrixBase(int n = 0) :
        length(n)
    {

    }

    inline
    int get_length() const
    {
        return length;
    }

};

template<typename ModeDefinition, typename T>
class Matrix: public MatrixBase<T>
{
  friend class Context<ModeDefinition>;
    friend class Kernel<ModeDefinition> ;
    friend class MatrixFactory<ModeDefinition, T> ;
    typedef typename ModeDefinition::Status Status;
    typedef typename ModeDefinition::DevicePtr_Handle DevicePtr_Handle;
    typedef typename ModeDefinition::Context_Handle Context_Handle;
    typedef typename ModeDefinition::Stream Stream;
    typedef MatrixBase<T> Base;
protected:
    Stream stream;
    Context_Handle cont;
    DevicePtr_Handle ptr;
    Matrix(Context_Handle, Stream, int n, T* tmp = NULL);
    Matrix();
    Matrix(const Matrix& other_matrix);

public:
    const Matrix& operator=(const Matrix& input);
    DevicePtr_Handle& get_ptr();
    void set_ptr(double*);
    ~Matrix();
    void to_cpu_ptr(T*) const;
//    inline void delete_cpu_ptr() const;
};

template<typename ModeDefinition, typename T>
class GLMatrix: public MatrixBase<T>
{
    typedef typename ModeDefinition::Context_Handle Context_Handle;
    typedef typename ModeDefinition::Stream Stream;
    typedef typename ModeDefinition::Status Status;
    typedef typename ModeDefinition::DevicePtr_Handle DevicePtr_Handle;
    typedef typename ModeDefinition::Graphics_Handle Graphics_Handle;
    typedef MatrixBase<T> Base;
protected:
    Graphics_Handle vbo;
    Stream stream;
public:
    GLMatrix(Context_Handle, Stream,GLuint ptr);
    DevicePtr_Handle map_ptr();
    void unmap();
};

#endif /* MATRIX_HPP_ */
