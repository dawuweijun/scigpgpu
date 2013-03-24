/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/

#ifndef SCI_MAKECUCOMPLEX_H_
#define SCI_MAKECUCOMPLEX_H_
#include <cuComplex.h>
#ifdef __cplusplus
extern "C"
{
#endif

    cudaError_t writecucomplex(double* h, double* hi, int rows, int cols, cuDoubleComplex* d_data);
    cudaError_t rewritecucomplex(double* d, int rows, int cols, cuDoubleComplex* d_data);
    cudaError_t createcucomplex(double* d, double* di, int rows, int cols, cuDoubleComplex* d_data);
    cudaError_t readcucomplex(double* h, double* hi, int rows, int cols, cuDoubleComplex* d_data);

#ifdef __cplusplus
};
#endif /* extern "C" */

#endif /* SCI_MAKECUCOMPLEX_H_ */
