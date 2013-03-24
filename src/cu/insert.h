/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) Scilab Enterprises - 2013 - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/
#ifndef _INSERT_H_
#define _INSERT_H_

#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif
    cudaError_t cudaInsert(double* d_inputA, int inputSize, double* d_output, double* pdblPos, int outputSize, int isScalar, int* piErr);
    cudaError_t cudaZInsert(cuDoubleComplex* d_inputA, int inputSize, cuDoubleComplex* d_output, double* pdblPos, int outputSize, int isScalar, int* piErr);
    cudaError_t cudaZDInsert(cuDoubleComplex* d_inputA, int inputSize, double* d_output, double* pdblPos, int outputSize, int isScalar, int* piErr);
#ifdef __cplusplus
}
#endif /* extern "C" */


#endif /* _INSERT_H_ */
