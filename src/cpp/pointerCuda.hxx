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

#ifndef __POINTERCUDA_HXX__
#define __POINTERCUDA_HXX__

#include "gpuPointer.hxx"
#include "gpuContext.hxx"
#include "dynlib_gpu.h"

#include <memory> // std::shared_ptr

class GPU_IMPEXP PointerCuda : public GpuPointer
{
public :
    PointerCuda();
    PointerCuda(int rows, int cols, bool complex); // allocate device memory
    PointerCuda(int dims, int* dimsArray, bool complex); // allocate device memory
    PointerCuda(double* ptr, int rows, int cols); // real data
    PointerCuda(double* ptr, int dims, int* dimsArray); // real data
    PointerCuda(double* ptrReal, double* ptrImg, int rows, int cols); // complex data
    PointerCuda(double* ptrReal, double* ptrImg, int dims, int* dimsArray); // complex data

    GpuPointer* operator+(GpuPointer &b);
    GpuPointer* operator*(GpuPointer &b);
    GpuPointer* operator-(GpuPointer &b);
    GpuPointer* FFT(int iSign, int* piDim, int iSizeDim, int* piIncr);
    GpuPointer* transpose();
    GpuPointer* clone();
    GpuPointer* extract(GpuPointer* gpuPos);
    int insert(GpuPointer* gpuData, GpuPointer* gpuPos, int isScalar);

    double getMax();
    double getMin();
    double getNorm();
    double getSum();
    cuDoubleComplex getComplexMax();
    cuDoubleComplex getComplexMin();
    cuDoubleComplex getComplexSum();

    void initMatrix(double);

    int getRows(void);
    int getCols(void);
    void setRows(int);
    void setCols(int);
    int getDims(void);
    int* getDimsArray(void);
    int getSize(void);
    int getSizeOfElem(void);
    bool isGpuComplex(void);
    void getData(double*); // get data on host
    void getData(double*, double*); // get complex data on host
    GpuType getGpuType(void);
    double* getGpuPtr(); // get the shared_ptr _ptr

    ~PointerCuda();

private :
    std::shared_ptr<Matrix<ModeDefinition<CUDA>, double> > _ptr;
    cudaError_t _cudaStat;
};

#endif //__POINTERCUDA_HXX__
