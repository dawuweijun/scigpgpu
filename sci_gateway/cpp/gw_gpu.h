/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2011 - Allan CORNET
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/
/* ==================================================================== */
#ifndef __GW_GPU_H__
#define __GW_GPU_H__

#ifdef __cplusplus
extern "C" {
#endif

    int sci_gpuInit(char* fname);
    int sci_gpuExit(char* fname);
    int sci_gpuDoubleCapability(char* fname);
    int sci_gpuAdd(char *fname);
    int sci_gpuFFT(char *fname);
    int sci_gpuMax(char *fname);
    int sci_gpuMin(char *fname);
    int sci_gpuNorm(char *fname);
    int sci_gpuMult(char *fname);
    int sci_gpuTranspose(char *fname);
    int sci_gpuSum(char *fname);
    int sci_gpuAlloc(char *fname);
    int sci_gpuApplyFunction(char *fname);
    int sci_gpuDeviceInfo(char *fname);
    int sci_gpuDeviceMemInfo(char *fname);
    int sci_gpuDoubleCapability(char *fname);
    int sci_gpuExit(char *fname);
    int sci_gpuFree(char *fname);
    int sci_gpuGetArgs(char *fname);
    int sci_gpuInit(char *fname);
    int sci_gpuLoadFunction(char *fname);
    int sci_gpuSize(char *fname);
    int sci_gpuSetData(char *fname);
    int sci_gpuGetData(char *fname);
    int sci_gpuBuild(char *fname);
    int sci_gpuUseCuda(char *fname);
    int sci_gpuPtrInfo(char *fname);
    int sci_isGpuPointer(char *fname);
    int sci_gpuInterp(char *fname);
    int sci_gpuInterp2d(char *fname);
    int sci_gpuMatrix(char *fname);
    int sci_gpuExtract(char *fname);
    int sci_gpuInsert(char *fname);
    int sci_gpuSubtract(char *fname);
    int sci_gpuClone(char *fname);
    int sci_gpuDotMult(char *fname);
    int sci_gpuComplex(char *fname);
    int sci_gpuSplin2d(char *fname);
    int sci_gpuKronecker(char *fname);
    int sci_gpuOnes(char *fname);

#ifdef __cplusplus
}
#endif /* extern "C" */

#endif /* __GW_GPU_H__ */
/* ==================================================================== */
