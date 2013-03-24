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
#include "interp.h"

// x(1..n) being an array (with strict increasing order and n >=2)
// representing intervals, this routine return i such that :
//
// x(i) <= t <= x(i+1)
//
// and -1 if t is not in [x(1), x(n)]
__device__ int isearch_gpu(double dXp, double* X, int sizeOfX)
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

__global__ void interp_kernel(  double* Xp, double* Yp, double* Yp1, double* Yp2, double* Yp3, int sizeOfXp,
                                double* X, double* Y, double* D, int sizeOfX, int iType)
{
    int iPos = blockIdx.x * blockDim.x + threadIdx.x;

    if(iPos >= sizeOfXp)
    {
        return;
    }

    double dXp = Xp[iPos];

    int i = isearch_gpu(dXp, X, sizeOfX);

    if(i == -1) // dXp is outside [X(1), X(n)] evaluation depend upon outmode (iType)
    {
        if(iType == 10 || isnan(dXp)) // BY_NAN
        {
            int iOne = 1;
            double nan = 1.0;
            nan = (nan - (double)iOne) / (nan - (double)iOne);

            Yp[iPos]  = nan;
            Yp1[iPos] = nan;
            Yp2[iPos] = nan;
            Yp3[iPos] = nan;
            return;
        }
        else if(iType == 7) // BY_ZERO
        {
            Yp[iPos]  = 0;
            Yp1[iPos] = 0;
            Yp2[iPos] = 0;
            Yp3[iPos] = 0;
            return;
        }
        else if(iType == 8) // C0
        {
            if(dXp < X[0])
            {
                Yp[iPos] = Y[0];
            }
            else
            {
                Yp[iPos] = Y[sizeOfX - 1];
            }

            Yp1[iPos] = 0;
            Yp2[iPos] = 0;
            Yp3[iPos] = 0;
            return;
        }
        else if(iType == 9) // LINEAR
        {
            if(dXp < X[0])
            {
                Yp[iPos]  = Y[0] + (dXp - X[0]) * D[0];
                Yp1[iPos] = D[0];
            }
            else
            {
                Yp[iPos]  = Y[sizeOfX - 1] + (dXp - X[sizeOfX - 1]) * D[sizeOfX - 1];
                Yp1[iPos] = D[sizeOfX - 1];
            }

            Yp2[iPos] = 0;
            Yp3[iPos] = 0;
            return;
        }
        else
        {
            if(iType == 1) // NATURAL
            {
                if(dXp < X[0])
                {
                    i = 0;
                }
                else
                {
                    i = sizeOfX - 2;
                }
            }
            else if(iType == 3) // PERIODIC
            {
                // recompute t such that t in [x(1), x(n)] by periodicity :
                // and then the interval i of this new t

                double dx = X[sizeOfX -1 ] - X[0];
                double r  = (dXp - X[0]) / dx;

                if(r >= 0)
                {
                    dXp = X[0] + (r - trunc(r)) * dx;
                }
                else
                {
                    r = abs(r);
                    dXp = X[sizeOfX - 1] - (r - trunc(r)) * dx;
                }

                // some cautions in case of roundoff errors (is necessary ?)
                if(dXp < X[0])
                {
                    dXp = X[0];
                    i = 0;
                }
                else if(dXp > X[sizeOfX - 1])
                {
                    dXp = X[sizeOfX - 1];
                    i = sizeOfX - 2;
                }
                else
                {
                    i = isearch_gpu(dXp, X, sizeOfX);
                }
            }
        }
    }

    double dX1 = X[i];
    double dX2 = X[i+1];
    double dY1 = Y[i];
    double dY2 = Y[i+1];
    double dD1 = D[i];
    double dD2 = D[i+1];

    //        compute the following Newton form :
    //        h(t) = dY1 + dD1*(t-dX1) + c2*(t-dX1)^2 + c3*(t-dX1)^2*(t-dX2)
    double dx = 1.0 / (dX2 - dX1);
    double p  = (dY2 - dY1) * dx;
    double c2 = (p - dD1) * dx;
    double c3 = ((dD2 - p) + (dD1 - p)) * (dx * dx);

    //        eval h(t), h'(t), h"(t) and h"'(t), by a generalised Horner 's scheme
    double tmxa = dXp - dX1;
    double h    = c2 + c3*(dXp - dX2);
    double dh   = h + c3 * tmxa;
    double ddh  = 2.0 * (dh + c3 * tmxa);
    double dddh = 6.0 * c3;

    h  = dD1 + h * tmxa;
    dh = h + dh * tmxa;
    h  = dY1 + h * tmxa;

    Yp[iPos]  = h;
    Yp1[iPos] = dh;
    Yp2[iPos] = ddh;
    Yp3[iPos] = dddh;

}

cudaError_t interp_gpu( double* Xp, double* Yp, double* Yp1, double* Yp2, double* Yp3, int sizeOfXp,
                        double* X, double* Y, double* D, int sizeOfX, int iType)
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

        interp_kernel<<<grid, block>>>(Xp, Yp, Yp1, Yp2, Yp3, sizeOfXp, X, Y, D, sizeOfX, iType);

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


