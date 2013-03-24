/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2012 - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/
#ifndef SCI_INTERP2D_H_
#define SCI_INTERP2D_H_

#ifdef __cplusplus
extern "C"{
#endif
cudaError_t interp2d_gpu(double* X, double* Y, double* C, int sizeOfX, int sizeOfY,
                         double* Xp, double* Yp, double* Zp, int sizeOfXp, int iType);

cudaError_t interp2dWithGrad_gpu(double* X, double* Y, double* C, int sizeOfX, int sizeOfY,
                                 double* Xp, double* Yp, double* Zp, double* dZdXp, double* dZdYp,
                                 int sizeOfXp, int iType);

cudaError_t interp2dWithGradAnHes_gpu(double* X, double* Y, double* C, int sizeOfX, int sizeOfY,
                                      double* Xp, double* Yp, double* Zp, double* dZdXp, double* dZdYp,
                                      double* d2Zd2Xp, double* d2ZdXYp, double* d2Zd2Yp,
                                      int sizeOfXp, int iType);
#ifdef __cplusplus
}
#endif /* extern "C" */


#endif /* SCI_INTERP2D_H_ */
