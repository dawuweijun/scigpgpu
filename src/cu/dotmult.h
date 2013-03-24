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
#ifndef SCI_DOTMULT_H_
#define SCI_DOTMULT_H_

#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif
    cudaError_t cudaDotMult(int elems, double* dA, double* dB, double* dRes);
    cudaError_t cudaZDotMult(int elems, cuDoubleComplex* dA, cuDoubleComplex* dB, cuDoubleComplex* dRes);
    cudaError_t cudaZDDotMult(int elems, cuDoubleComplex* dA, double* dB, cuDoubleComplex* dRes);
#ifdef __cplusplus
}
#endif /* extern "C" */


#endif /* SCI_DOTMULT_H_ */
