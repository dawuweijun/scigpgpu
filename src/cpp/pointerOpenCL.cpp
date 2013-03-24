/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2011 - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/

#include "pointerOpenCL.hxx"
#include "gpuContext.hxx"

PointerOpenCL::PointerOpenCL()
{
}

PointerOpenCL::PointerOpenCL(int rows, int cols, bool complex)
{
    _ptr = getOpenClContext()->genMatrix<double>(getOpenClQueue(), rows * cols);
    _bComplex    = complex;
    _iCols       = cols;
    _iRows       = rows;
    _iTotalSize  = rows * cols;
}

PointerOpenCL::PointerOpenCL(int dims, int* dimsArray, bool complex)
{
    int iTotalSize = 1;
    for ( int i = 0; i < dims; i++)
    {
        iTotalSize *= dimsArray[i];
    }

    if (complex)
    {
        _ptr = getOpenClContext()->genMatrix<double>(getOpenClQueue(), iTotalSize * 2);
    }
    else
    {
        _ptr = getOpenClContext()->genMatrix<double>(getOpenClQueue(), iTotalSize);
    }

    _bComplex   = complex;
    _iRows      = dimsArray[0];
    _iCols      = 1;

    if (dims > 1)
    {
        _iCols = dimsArray[1];
    }

    _iDims = dims;
    _iDimsArray = dimsArray;
    _iTotalSize = iTotalSize;
}

PointerOpenCL::PointerOpenCL(double* data, int rows, int cols)
{
    _ptr = getOpenClContext()->genMatrix<double>(getOpenClQueue(), rows * cols, data);

    _bComplex    = false;
    _iCols       = cols;
    _iRows       = rows;
    _iTotalSize  = rows * cols;
}

PointerOpenCL::PointerOpenCL(double* data, int dims, int* dimsArray)
{
    int iTotalSize = 1;
    for ( int i = 0; i < dims; i++)
    {
        iTotalSize *= dimsArray[i];
    }

    _ptr = getOpenClContext()->genMatrix<double>(getOpenClQueue(), iTotalSize, data);
    _bComplex   = false;
    _iRows      = dimsArray[0];
    _iCols      = 1;

    if (dims > 1)
    {
        _iCols = dimsArray[1];
    }

    _iDims = dims;
    _iDimsArray = dimsArray;
    _iTotalSize = iTotalSize;
}

int PointerOpenCL::getRows(void)
{
    return _iRows;
}
int PointerOpenCL::getCols(void)
{
    return _iCols;
}
int PointerOpenCL::getDims(void)
{
    return _iDims;
}
int* PointerOpenCL::getDimsArray(void)
{
    return _iDimsArray;
}
int PointerOpenCL::getSize(void)
{
    return _iTotalSize;
}
bool PointerOpenCL::isGpuComplex(void)
{
    return _bComplex;
}
GpuPointer::GpuType PointerOpenCL::getGpuType(void)
{
    return OpenCLType;
}
double* PointerOpenCL::getGpuPtr(void)
{
    return (double*)_ptr->get_ptr();
}
void PointerOpenCL::getData(double* d)
{
    _ptr->to_cpu_ptr(d);
}
PointerOpenCL::~PointerOpenCL()
{
    _ptr.reset();
}

