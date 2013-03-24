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
#ifndef _ELEMWISEMIN_H_
#define _ELEMWISEMIN_H_

#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif
    cudaError_t cudaMinElementwise(double* d_inputA, double* d_inputB, double* d_output, int rows, int cols);
    cudaError_t cudaZMinElementwise(cuDoubleComplex* d_inputA, cuDoubleComplex* d_inputB, cuDoubleComplex* d_output, int rows, int cols);
    cudaError_t cudaZDMinElementwise(cuDoubleComplex* d_inputA, double* d_inputB, cuDoubleComplex* d_output, int rows, int cols);
#ifdef __cplusplus
}
#endif /* extern "C" */


#endif /* _ELEMWISEMIN_H_ */
