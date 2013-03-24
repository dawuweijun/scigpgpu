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

#ifndef __POINTEROPENCL_HXX__
#define __POINTEROPENCL_HXX__

#include "gpuPointer.hxx"
#include "gpuContext.hxx"
#include "dynlib_gpu.h"

#include <memory> // std::shared_ptr

class GPU_IMPEXP PointerOpenCL : public GpuPointer
{
public :
    PointerOpenCL();
    PointerOpenCL(int rows, int cols, bool complex);
    PointerOpenCL(int dims, int* dimsArray, bool complex);
    PointerOpenCL(double* ptr, int rows, int cols);
    PointerOpenCL(double* ptr, int dims, int* dimsArray);

    int getRows(void);
    int getCols(void);
    int getDims(void);
    int* getDimsArray(void);
    int getSize(void);
    bool isGpuComplex(void);
    void getData(double*);
    GpuType getGpuType(void);
    double* getGpuPtr();

    ~PointerOpenCL();

private :
    std::shared_ptr<Matrix<ModeDefinition<OpenCL>, double> > _ptr;
};

#endif //__POINTEROPENCL_HXX__
