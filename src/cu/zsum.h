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
#ifndef SCI_ZSUM_H_
#define SCI_ZSUM_H_

#include <cuComplex.h>
#ifdef __cplusplus
extern "C"{
#endif
    cudaError_t cudaZsum(int elems, cuDoubleComplex* d, cuDoubleComplex* res);
#ifdef __cplusplus
}
#endif /* extern "C" */

#endif /* SCI_ZSUM_H_ */
