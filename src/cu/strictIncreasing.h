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
#ifndef _STRICTINCREASING_H_
#define _STRICTINCREASING_H_

#ifdef __cplusplus
extern "C" {
#endif
    cudaError_t cudaStrictIncreasing(double* d_input, int iSize, int* isStrictIncreasing);
#ifdef __cplusplus
}
#endif /* extern "C" */


#endif /* _STRICTINCREASING_H_ */
