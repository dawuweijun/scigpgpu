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

#ifndef SCI_KRONECKER_H_
#define SCI_KRONECKER_H_

#ifdef __cplusplus
extern "C"
{
#endif
    // dOut = dA .*. dB
    cudaError_t cudaKronecker(double* dA, int iRowsA, int iColsA, bool isAComplex, double* dB, int iRowsB, int iColsB, bool isBComplex, double* dOut);

#ifdef __cplusplus
};
#endif /* extern "C" */

#endif /* SCI_KRONECKER_H_*/
