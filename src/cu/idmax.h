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
#ifndef SCI_IDMAX_H_
#define SCI_IDMAX_H_

#ifdef __cplusplus
extern "C"{
#endif
    cudaError_t cudaIdmax(int elems, double* d, double* res);
#ifdef __cplusplus
}
#endif /* extern "C" */


#endif /* SCI_IDMAX_H_ */
