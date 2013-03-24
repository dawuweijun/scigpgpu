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

#include <math.h>
#include <stdio.h>
#include "interp2d.h"

/*****************************************/
/************ Device funtions ************/
/*****************************************/
// x(1..n) being an array (with strict increasing order and n >=2)
// representing intervals, this routine return i such that :
//
// x(i) <= t <= x(i+1)
//
// and -1 if t is not in [x(1), x(n)]
__device__ int isearch2d_gpu(double dXp, double* X, int sizeOfX)
{
    if(X[0] <= dXp && dXp <= X[sizeOfX - 1])
    { // dichotomic search
        int i  = 0;
        int i1 = 0;
        int i2 = sizeOfX - 1;

        while(i2 - i1 > 1)
        {
            i = (i1 + i2) / 2;
            if(dXp <= X[i])
            {
                i2 = i;
            }
            else
            {
                i1 = i;
            }
        }

        return i1;
    }

    return -1;
}

__device__ int coord_by_periodicity(double* dXp, double* X, int sizeOfX)
{
    int i = -1;

    // recompute t such that t in [x(1), x(n)] by periodicity :
    // and then the interval i of this new t

    double dx = X[sizeOfX - 1] - X[0];
    double r  = (*dXp - X[0]) / dx;

    if(r >= 0)
    {
        *dXp = X[0] + (r - trunc(r)) * dx;
    }
    else
    {
        r = abs(r);
        *dXp = X[sizeOfX - 1] - (r - trunc(r)) * dx;
    }

    // some cautions in case of roundoff errors (is necessary ?)
    if(*dXp < X[0])
    {
        *dXp = X[0];
        i = 0;
    }
    else if(*dXp > X[sizeOfX - 1])
    {
        *dXp = X[sizeOfX - 1];
        i = sizeOfX - 2;
    }
    else
    {
        i = isearch2d_gpu(*dXp, X, sizeOfX);
    }

    return i;
}

__device__ int iEdge_gpu(double dXp, double* X, int sizeOfX)
{
    int i = 0;

    if(dXp < X[0])
    {
        i = 0;
    }
    else
    {
        i = sizeOfX - 2;
    }

    return i;
}

__device__ double bicubicInterp(double dx, double dy, double* pdblC)
{
    //Bicubic interpolation
    double dZp = 0;
    dZp = pdblC[3] + dy * (pdblC[4 + 3] + dy * (pdblC[8 + 3] + dy * pdblC[12 + 3]));
    dZp = pdblC[2] + dy * (pdblC[4 + 2] + dy * (pdblC[8 + 2] + dy * pdblC[12 + 2])) + dZp * dx;
    dZp = pdblC[1] + dy * (pdblC[4 + 1] + dy * (pdblC[8 + 1] + dy * pdblC[12 + 1])) + dZp * dx;
    dZp = pdblC[0] + dy * (pdblC[4 + 0] + dy * (pdblC[8 + 0] + dy * pdblC[12 + 0])) + dZp * dx;
    return dZp;

}

__device__ void bicubicInterpWithGrad(double dx, double dy, double* pdblC, bool computeX, bool computeY, double* results)
{
    //Bicubic interpolation
    double dZdXp = 0;
    double dZdYp = 0;

    if(computeX)
    {
        dZdXp = pdblC[3 * 4 + 1] + dx * (2 * pdblC[3 * 4 + 2] + dx * 3 * pdblC[3 * 4 + 3]);
        dZdXp = pdblC[2 * 4 + 1] + dx * (2 * pdblC[2 * 4 + 2] + dx * 3 * pdblC[2 * 4 + 3]) + dZdXp * dy;
        dZdXp = pdblC[1 * 4 + 1] + dx * (2 * pdblC[1 * 4 + 2] + dx * 3 * pdblC[1 * 4 + 3]) + dZdXp * dy;
        dZdXp = pdblC[0 * 4 + 1] + dx * (2 * pdblC[0 * 4 + 2] + dx * 3 * pdblC[0 * 4 + 3]) + dZdXp * dy;
    }

    if(computeY)
    {
        dZdYp = pdblC[4 + 3] + dy * (2 * pdblC[8 + 3] + dy * 3 * pdblC[12 + 3]);
        dZdYp = pdblC[4 + 2] + dy * (2 * pdblC[8 + 2] + dy * 3 * pdblC[12 + 2]) + dZdYp * dx;
        dZdYp = pdblC[4 + 1] + dy * (2 * pdblC[8 + 1] + dy * 3 * pdblC[12 + 1]) + dZdYp * dx;
        dZdYp = pdblC[4 + 0] + dy * (2 * pdblC[8 + 0] + dy * 3 * pdblC[12 + 0]) + dZdYp * dx;
    }

    results[0] = dZdXp;
    results[1] = dZdYp;

}

__device__ void bicubicInterpWithGradAndHes(double dx, double dy, double* pdblC, bool computeX, bool computeY, double* results)
{
    //Bicubic interpolation
    double dd2Zd2Xp = 0;
    double dd2ZdXYp = 0;
    double dd2Zd2Yp = 0;

    if(computeX)
    {
        dd2Zd2Xp = 2 * pdblC[3 * 4 + 2] + dx * 6 * pdblC[3 * 4 + 3];
        dd2Zd2Xp = 2 * pdblC[2 * 4 + 2] + dx * 6 * pdblC[2 * 4 + 3] + dd2Zd2Xp * dy;
        dd2Zd2Xp = 2 * pdblC[1 * 4 + 2] + dx * 6 * pdblC[1 * 4 + 3] + dd2Zd2Xp * dy;
        dd2Zd2Xp = 2 * pdblC[0 * 4 + 2] + dx * 6 * pdblC[0 * 4 + 3] + dd2Zd2Xp * dy;
    }

    if(computeY)
    {
        dd2Zd2Yp = 2 * pdblC[8 + 3] + dy * 6 * pdblC[12 + 3];
        dd2Zd2Yp = 2 * pdblC[8 + 2] + dy * 6 * pdblC[12 + 2] + dd2Zd2Yp * dx;
        dd2Zd2Yp = 2 * pdblC[8 + 1] + dy * 6 * pdblC[12 + 1] + dd2Zd2Yp * dx;
        dd2Zd2Yp = 2 * pdblC[8 + 0] + dy * 6 * pdblC[12 + 0] + dd2Zd2Yp * dx;
    }

    if(computeX && computeY)
    {
        dd2ZdXYp    =            pdblC[4 + 1] + dy * (2 * pdblC[8 + 1] + dy * 3 * pdblC[12 + 1])
                    + dx * (2 * (pdblC[4 + 2] + dy * (2 * pdblC[8 + 2] + dy * 3 * pdblC[12 + 2]))
                    + dx * (3 * (pdblC[4 + 3] + dy * (2 * pdblC[8 + 3] + dy * 3 * pdblC[12 + 3]))));
    }

    results[0] = dd2Zd2Xp;
    results[1] = dd2ZdXYp;
    results[2] = dd2Zd2Yp;
}

/**********************************************/
/************ Interpolation Kernel ************/
/**********************************************/
__global__ void interp2d_kernel(double* X, double* Y, double* C, int sizeOfX, int sizeOfY,
                                double* Xp, double* Yp, double* Zp, double* ZdXp, double* ZdYp,
                                double* d2Zd2Xp, double* d2ZdXYp, double* d2Zd2Yp,
                                int sizeOfXp, int iType)
{
    int iPos = blockIdx.x * blockDim.x + threadIdx.x;

    if(iPos >= sizeOfXp)
    {
        return;
    }

    double dXp = Xp[iPos];
    double dYp = Yp[iPos];

    bool bicubicWithGrad = ZdXp ? true : false;
    bool bicubicWithGradAndHes = d2Zd2Xp ? true : false;

    bool compute_dzdx = true;
    bool compute_dzdy = true;

    int i = isearch2d_gpu(dXp, X, sizeOfX);
    int j = isearch2d_gpu(dYp, Y, sizeOfY);

    if(i == -1 || j == -1) // dXp and dYp were outside [X(1), X(n)] evaluation depend upon outmode (iType)
    {
        if(iType == 10 || isnan(dXp) || isnan(dYp)) // BY_NAN
        {
            int iOne = 1;
            double nan = 1.0;
            nan = (nan - (double)iOne) / (nan - (double)iOne);

            Zp[iPos] = nan;

            if(bicubicWithGrad)
            {
                ZdXp[iPos] = nan;
                ZdYp[iPos] = nan;
            }

            if(bicubicWithGradAndHes)
            {
                d2Zd2Xp[iPos] = nan;
                d2ZdXYp[iPos] = nan;
                d2Zd2Yp[iPos] = nan;
            }

            return;
        }
        else if(iType == 7) // BY_ZERO
        {
            Zp[iPos] = 0;

            if(bicubicWithGrad)
            {
                ZdXp[iPos] = 0;
                ZdYp[iPos] = 0;
            }

            if(bicubicWithGradAndHes)
            {
                d2Zd2Xp[iPos] = 0;
                d2ZdXYp[iPos] = 0;
                d2Zd2Yp[iPos] = 0;
            }

            return;
        }
        else if(iType == 8) // C0
        {
            if(i == -1)
            {
                i = iEdge_gpu(dXp, X, sizeOfX);
                dXp = i ? X[i+1] : X[0];
                compute_dzdx = false;
            }

            if(j == -1)
            {
                j = iEdge_gpu(dYp, Y, sizeOfY);
                dYp = j ? Y[j+1] : Y[0];
                compute_dzdy = false;
            }
        }
        else if(iType == 1) // NATURAL
        {
            if(i == -1)
            {
                i = iEdge_gpu(dXp, X, sizeOfX);
            }

            if(j == -1)
            {
                j = iEdge_gpu(dYp, Y, sizeOfY);
            }
        }
        else if(iType == 3) // PERIODIC
        {
            if(i == -1)
            {
                i = coord_by_periodicity(&dXp, X, sizeOfX);
            }

            if(j == -1)
            {
                j = coord_by_periodicity(&dYp, Y, sizeOfY);
            }
        }
    }

    double dx = dXp - X[i];
    double dy = dYp - Y[j];
    double* pdblC = C + (j * (sizeOfX - 1) + i) * 16;

    Zp[iPos] = bicubicInterp(dx, dy, pdblC);

    if(bicubicWithGrad) // Bicubic interpolation with gradient
    {
        double pdblResWithGrad[2];
        bicubicInterpWithGrad(dx, dy, pdblC, compute_dzdx, compute_dzdy, pdblResWithGrad);
        ZdXp[iPos] = pdblResWithGrad[0];
        ZdYp[iPos] = pdblResWithGrad[1];
    }

    if(bicubicWithGradAndHes)
    {
        double pdblResWithGradAndHes[3];
        bicubicInterpWithGradAndHes(dx, dy, pdblC, compute_dzdx, compute_dzdy, pdblResWithGradAndHes);
        d2Zd2Xp[iPos] = pdblResWithGradAndHes[0];
        d2ZdXYp[iPos] = pdblResWithGradAndHes[1];
        d2Zd2Yp[iPos] = pdblResWithGradAndHes[2];
    }
}

/***********************************************/
/************ Bicubic interpolation ************/
/***********************************************/
cudaError_t interp2d_gpu(double* X, double* Y, double* C, int sizeOfX, int sizeOfY,
                         double* Xp, double* Yp, double* Zp, int sizeOfXp, int iType)
{

    cudaError_t cudaStat = cudaGetLastError();

    try
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        int threadMax = deviceProp.maxThreadsDim[0];
//        int blockMax  = deviceProp.maxGridSize[0];

        int iDimBlock = (sizeOfXp > threadMax) ? threadMax : sizeOfXp;
        int iDimGrid  = (int)ceil((float)sizeOfXp/iDimBlock);

        dim3 block(iDimBlock, 1, 1);
        dim3 grid(iDimGrid, 1, 1);

        interp2d_kernel<<<grid, block>>>(X, Y, C, sizeOfX, sizeOfY, Xp, Yp, Zp, NULL, NULL,
                                         NULL, NULL, NULL, sizeOfXp, iType);

        cudaStat = cudaThreadSynchronize();
        if (cudaStat != cudaSuccess) throw cudaStat;

        return cudaSuccess;
    }
    catch(cudaError_t cudaE)
	{
		return cudaE;
	}
}

/**************************************************************/
/************ Bicubic interpolation with gradient ************/
/**************************************************************/
cudaError_t interp2dWithGrad_gpu(double* X, double* Y, double* C, int sizeOfX, int sizeOfY,
                                 double* Xp, double* Yp, double* Zp, double* dZdXp, double* dZdYp,
                                 int sizeOfXp, int iType)
{
    cudaError_t cudaStat = cudaGetLastError();

    try
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        int threadMax = deviceProp.maxThreadsDim[0];
//        int blockMax  = deviceProp.maxGridSize[0];

        int iDimBlock = (sizeOfXp > threadMax) ? threadMax : sizeOfXp;
        int iDimGrid  = (int)ceil((float)sizeOfXp/iDimBlock);

        dim3 block(iDimBlock, 1, 1);
        dim3 grid(iDimGrid, 1, 1);

        interp2d_kernel<<<grid, block>>>(X, Y, C, sizeOfX, sizeOfY, Xp, Yp, Zp, dZdXp, dZdYp,
                                         NULL, NULL, NULL, sizeOfXp, iType);

        cudaStat = cudaThreadSynchronize();
        if (cudaStat != cudaSuccess) throw cudaStat;

        return cudaSuccess;
    }
    catch(cudaError_t cudaE)
    {
        return cudaE;
    }
}

/**************************************************************************/
/************ Bicubic interpolation with gradient and Hessean ************/
/**************************************************************************/
cudaError_t interp2dWithGradAnHes_gpu(double* X, double* Y, double* C, int sizeOfX, int sizeOfY,
                                      double* Xp, double* Yp, double* Zp, double* dZdXp, double* dZdYp,
                                      double* d2Zd2Xp, double* d2ZdXYp, double* d2Zd2Yp,
                                      int sizeOfXp, int iType)
{
    cudaError_t cudaStat = cudaGetLastError();

    try
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        int threadMax = deviceProp.maxThreadsDim[0];
//        int blockMax  = deviceProp.maxGridSize[0];

        int iDimBlock = (sizeOfXp > threadMax) ? threadMax : sizeOfXp;
        int iDimGrid  = (int)ceil((float)sizeOfXp/iDimBlock);

        dim3 block(iDimBlock, 1, 1);
        dim3 grid(iDimGrid, 1, 1);

        interp2d_kernel<<<grid, block>>>(X, Y, C, sizeOfX, sizeOfY, Xp, Yp, Zp, dZdXp, dZdYp,
                                         d2Zd2Xp, d2ZdXYp, d2Zd2Yp, sizeOfXp, iType);

        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        cudaStat = cudaThreadSynchronize();
        if (cudaStat != cudaSuccess) throw cudaStat;

        return cudaSuccess;
    }
    catch(cudaError_t cudaE)
    {
        return cudaE;
    }
}


