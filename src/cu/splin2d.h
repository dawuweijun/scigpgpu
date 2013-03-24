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
#ifndef _CUDASPLIN2D_H_
#define _CUDASPLIN2D_H_

#include "splinType.h"

#ifdef __cplusplus
extern "C" {
#endif
    cudaError_t cudaBicubicSubSplin(double* X, double* Y, double* Z, int sizeOfX, int sizeOfY,
                                    double* P, double* Q, double* R, SplineType spType, double* C);

    cudaError_t cudaBicubicSplin(double* X, double* Y, double* Z, int sizeOfX, int sizeOfY,
                                 double* P, double* Q, double* R,
                                 double* Ad, double* Asd, double* Qdu,
                                 SplineType spType, double* C);

#ifdef __cplusplus
}
#endif /* extern "C" */


#endif /* _CUDASPLIN2D_H_ */
