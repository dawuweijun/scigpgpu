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

#ifndef __GPUPOINTER_HXX__
#define __GPUPOINTER_HXX__

#include <string>
#include "dynlib_gpu.h"

class GPU_IMPEXP  GpuPointer
{
protected :
    GpuPointer() {};

public :
    enum GpuType {CudaType, OpenCLType};
    virtual GpuPointer* operator+(GpuPointer &b)
    {
        return NULL;
    }
    virtual GpuPointer* operator*(GpuPointer &b)
    {
        return NULL;
    }
    virtual GpuPointer* operator-(GpuPointer &b)
    {
        return NULL;
    }
    virtual GpuPointer* FFT(int iSign, int* piDim, int iSizeDim, int* piIncr)
    {
        return NULL;
    }
    virtual GpuPointer* transpose()
    {
        return NULL;
    }
    virtual GpuPointer* clone()
    {
        return NULL;
    }
    virtual GpuPointer* extract(GpuPointer* gpuPos)
    {
        return NULL;
    }
    virtual int insert(GpuPointer* gpuData, GpuPointer* gpuPos, int isScalar)
    {
        return 0;
    }

    virtual double getMax()
    {
        return 0;
    }
    virtual double getMin()
    {
        return 0;
    }
    virtual double getNorm()
    {
        return 0;
    }
    virtual double getSum()
    {
        return 0;
    }

    virtual void initMatrix(double) {};

    virtual void setRows(int) {};
    virtual void setCols(int) {};
    virtual int getSizeOfElem(void)
    {
        return 0;
    }
    virtual void getData(double*, double*) {}; // get complex data on host
    virtual double* getGpuPtr()
    {
        return NULL;   // get the shared_ptr _ptr
    }

    virtual int getRows(void)           = 0;
    virtual int getCols(void)           = 0;
    virtual int getDims(void)           = 0;
    virtual int* getDimsArray(void)     = 0;
    virtual int getSize(void)           = 0;
    virtual bool isGpuComplex(void)     = 0;
    virtual GpuType getGpuType(void)    = 0;
    virtual void getData(double*)       = 0;

    virtual ~GpuPointer() {};

protected :
    int   _iRows;
    int   _iCols;
    int   _iDims;
    int   _iTotalSize;
    int*  _iDimsArray;
    bool  _bComplex;
};

#endif //__GPUPOINTER_HXX__
