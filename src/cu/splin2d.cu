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

#include <math.h>
#include <stdio.h>
#include "splin2d.h"

/*****************************************/
/************ Device funtions ************/
/*****************************************/
// Sign-Testing Routine.
// Returns:
//    -1. if ARG1 and ARG2 are of opposite sign.
//     0. if either argument is zero.
//    +1. if ARG1 and ARG2 are of the same sign.
__device__ int dpchst(double arg1, double arg2)
{
    if(arg1 == 0 || arg2 == 0)
    {
        return 0;
    }

    return (arg1 * arg2) < 0 ? -1 : 1;
}
__device__ int computeCPos(int x, int y, int X, int Y, int sizeOfX)
{
    // C(4,4,sizeOfX-1,sizeOfY-1)
    // C(x,y,X,Y)
    return x + y*4 + 16*X + 16*(sizeOfX-1)*Y;
}
__device__ void triDiagLDLSolve(double* Ad, double* Asd, double* out, int sizeOfIn, int incr)
{
    for (int i = incr; i < sizeOfIn*incr; i+=incr)
    {
        double temp = Asd[i-incr];
        Asd[i-incr] = Asd[i-incr] / Ad[i-incr];
        Ad[i] = Ad[i] - temp * Asd[i-incr];
        out[i] = out[i] - Asd[i-incr] * out[i-incr];
    }

    out[(sizeOfIn-1)*incr] = out[(sizeOfIn-1)*incr] / Ad[(sizeOfIn-1)*incr];

    for (int i = (sizeOfIn-2)*incr; i >= 0; i-=incr)
    {
        out[i] = out[i] / Ad[i] - Asd[i] * out[i+incr];
    }
}
__device__ void cyclicTriDiagLDLtSolve(double* Ad, double* Asd, double* Qdu, double* out, int sizeOfIn, int incr)
{
    double temp1 = 0;
    double temp2 = 0;

    //compute the LDL^t factorization
    for (int i = 0; i < (sizeOfIn-2)*incr; i+=incr)
    {
        temp1 = Asd[i];
        temp2 = Qdu[i];
        Asd[i] = Asd[i] / Ad[i]; //elimination coef L(i,i-1)
        Qdu[i] = Qdu[i] / Ad[i]; //elimination coef L(i,i-1)

        Ad[i+incr] = Ad[i+incr] - Asd[i] * temp1; // elimination on line i+1
        Qdu[i+incr] = Qdu[i+incr] - Qdu[i] * temp1; // elimination on line n
        Ad[(sizeOfIn-1)*incr] = Ad[(sizeOfIn-1)*incr] - Qdu[i] * temp2; // elimination on line n
    }

    temp2 = Qdu[(sizeOfIn-2)*incr];
    Qdu[(sizeOfIn-2)*incr] = Qdu[(sizeOfIn-2)*incr] / Ad[(sizeOfIn-2)*incr];
    Ad[(sizeOfIn-1)*incr] = Ad[(sizeOfIn-1)*incr] - Qdu[(sizeOfIn-2)*incr] * temp2;

    // solve LDL^t x = b  (but use b for x and for the intermediary vectors...)
    for(int i = incr; i < (sizeOfIn-1)*incr; i+=incr)
    {
        out[i] = out[i] - Asd[i-incr] * out[i-incr];
    }

    for(int i = 0; i < (sizeOfIn-1)*incr; i+=incr)
    {
        out[(sizeOfIn-1) * incr] = out[(sizeOfIn-1) * incr] - Qdu[i] * out[i];
    }

    for(int i = 0; i < sizeOfIn*incr; i+=incr)
    {
        out[i] = out[i] / Ad[i];
    }

    out[(sizeOfIn-2) * incr] = out[(sizeOfIn-2) * incr] - Qdu[(sizeOfIn-2)*incr] * out[(sizeOfIn-1) * incr];
    for (int i = (sizeOfIn-3)*incr; i >= 0; i-=incr)
    {
        out[i] = out[i] - Asd[i] * out[i+incr] - Qdu[i]*out[(sizeOfIn-1) * incr];
    }
}

/**************************************/
/************ splin Kernel ************/
/**************************************/
__global__ void derivd_FP_kernel(int sizeOfIn, int iSize, double* In, double* Z, double* Out, int incr)
{// Fast Periodic
    int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
    int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
    int iPos  = iPosY * sizeOfIn + iPosX;

    if(iPosX >= iSize || iPosY >= iSize )
    {
        return;
    }

    double* z = Z + iPos;
    double* out = Out + iPos;

    if(sizeOfIn == 2)
    {
        out[0] = (z[incr] - z[0]) / (In[1] - In[0]);
        out[incr] = out[0];
        return;
    }

    double dx_r = In[sizeOfIn-1] - In[sizeOfIn-2];
    double du_r = (z[0] - z[(sizeOfIn-2) * incr]) / dx_r;

    for (int i = 0; i < sizeOfIn - 1; i++)
    {
        double dx_l = dx_r;
        double du_l = du_r;
        dx_r = In[i+1] - In[i];
        du_r = (z[(i+1) * incr] - z[i*incr]) / dx_r;
        double w_l = dx_r / (dx_l + dx_r);
        double w_r = 1 - w_l;
        out[i*incr] = w_l * du_l + w_r * du_r;
    }

    out[(sizeOfIn-1) * incr] = out[0];
}

__global__ void derivd_F_kernel(int sizeOfIn, int iSize, double* In, double* Z, double* Out, int incr)
{// Fast
    int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
    int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
    int iPos  = iPosY * sizeOfIn + iPosX;

    if(iPosX >= iSize || iPosY >= iSize )
    {
        return;
    }

    double* z = Z + iPos;
    double* out = Out + iPos;

    if(sizeOfIn == 2)
    {
        out[0] = (z[incr] - z[0]) / (In[1] - In[0]);
        out[incr] = out[0];
        return;
    }

    double dx_l = In[1] - In[0];
    double du_l = (z[incr] - z[0]) / dx_l;
    double dx_r = In[2] - In[1];
    double du_r = (z[2*incr] - z[incr]) / dx_r;
    double w_l = dx_r / (dx_l + dx_r);
    double w_r = 1 - w_l;
    out[0] = (1 + w_r) * du_l - w_r * du_r;
    out[incr] = w_l * du_l + w_r * du_r;

    for (int i = 2; i < sizeOfIn - 1; i++)
    {
        dx_l = dx_r;
        du_l = du_r;
        dx_r = In[i+1] - In[i];
        du_r = (z[(i+1)*incr] - z[i*incr]) / dx_r;
        w_l = dx_r / (dx_l + dx_r);
        w_r = 1 - w_l;
        out[i*incr] = w_l * du_l + w_r * du_r;
    }

    out[(sizeOfIn-1) * incr] = (1 + w_l) * du_r - w_l * du_l;
}

__global__ void dpchim_kernel(int sizeOfIn, int iSize, double* In, double* Z, double* Out, int incr)
{// Monotone
    int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
    int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
    int iPos  = iPosY * sizeOfIn + iPosX;

    if(iPosX >= iSize || iPosY >= iSize )
    {
        return;
    }

    double* z = Z + iPos;
    double* out = Out + iPos;

    double h1 = In[1] - In[0];
    double del1 = (z[incr] - z[0]) / h1;
    // case sizeOfIn = 2, use linear interpolation
    if(sizeOfIn == 2)
    {
        out[0] = del1;
        out[(sizeOfIn-1) * incr] = del1;
        return;
    }

    // normal case
    double h2 = In[2] - In[1];
    double del2 = (z[2 * incr] - z[incr]) / h2;

    // set out[0] via non-centered three-point formula,
    // adjusted to be shape-preserving
    double hsum = h1 + h2;
    double w1 = (h1 + hsum) / hsum;
    double w2 = -h1 / hsum;
    out[0] = w1 * del1 + w2 * del2;
    if (dpchst(out[0], del1) <= 0)
    {
        out[0] = 0;
    }
    else if(dpchst(del1, del2) < 0)
    {
        double dmax = 3 * del1;
        if(abs(out[0]) > abs(dmax))
        {
            out[0] = dmax;
        }
    }

    // loop through interior points
    for(int i = 1; i < sizeOfIn - 1; i++)
    {
        if (i != 1)
        {
            h1 = h2;
            h2 = In[i+1] - In[i];
            hsum = h1 + h2;
            del1 = del2;
            del2 = (z[(i+1)*incr] - z[i*incr]) / h2;
        }

        // set out[i] = 0 unless data are strictly monotonic
        out[i * incr] = 0;
        if(dpchst(del1, del2) > 0)
        {
            double hsumt3 = 3*hsum;
            w1 = (hsum + h1) / hsumt3;
            w2 = (hsum + h2) / hsumt3;
            double absdel1 = abs(del1);
            double absdel2 = abs(del2);
            double m = max(absdel1, absdel2);
            double drat1 = del1 / m;
            double drat2 = del2 / m;
            out[i*incr] = min(absdel1, absdel2) / (w1 * drat1 + w2 * drat2);
        }
    }

    w1 = -h2 / hsum;
    w2 = (h2 + hsum) / hsum;
    iPos = (sizeOfIn-1)*incr;
    out[iPos] = w1 * del1 + w2 * del2;
    if (dpchst(out[iPos], del2) <= 0)
    {
        out[iPos] = 0;
    }
    else if(dpchst(del1, del2) < 0)
    {
        double dmax = 3 * del2;
        if(abs(out[iPos]) > abs(dmax))
        {
            out[iPos] = dmax;
        }
    }
}

__global__ void splinCub_kernel(int sizeOfIn, int iSize, double* In, double* Z, double* Out,
                                double* Ad, double* Asd, double* Qdu, //double* ll,
                                int incr, SplineType spType)
{// natural, not_a_knot, clamped, periodic
    int iPosX = blockIdx.x * blockDim.x + threadIdx.x;
    int iPosY = blockIdx.y * blockDim.y + threadIdx.y;
    int iPos  = iPosY * sizeOfIn + iPosX;

    if(iPosX >= iSize || iPosY >= iSize)
    {
        return;
    }

    double* z = Z + iPos;
    double* out = Out + iPos;
    double* ad = Ad + iPos;
    double* asd = Asd + iPos;
    double* qdu = Qdu + iPos;

    if(sizeOfIn == 2)
    {
        if (spType != CLAMPED)
        {
            out[0] = (z[incr] - z[0]) / (In[1] - In[0]);
            out[incr] = out[0];
        }
        return;
    }

    for(int i = 0; i < sizeOfIn-1; i++)
    {
        asd[i*incr] = 1 / (In[i+1] - In[i]);
        qdu[i*incr] = (z[(i+1) * incr] - z[i*incr]) * asd[i*incr] * asd[i*incr];
    }

    for(int i = 1; i < sizeOfIn-1; i++)
    {
        ad[i*incr] = 2 * (asd[(i-1)*incr] + asd[i*incr]);
        out[i*incr] = 3 * (qdu[(i-1)*incr] + qdu[i*incr]);
    }

    if(spType == NATURAL)
    {
        ad[0]  = 2 * asd[0];
        out[0] = 3 * qdu[0];
        ad[(sizeOfIn-1)*incr] = 2 * asd[(sizeOfIn-2)*incr];
        out[(sizeOfIn-1)*incr] = 3 * qdu[(sizeOfIn-2)*incr];
        triDiagLDLSolve(ad, asd, out, sizeOfIn, incr);
    }
    else if(spType == NOT_A_KNOT)
    {
        // s'''(x(2)-) = s'''(x(2)+)
        double r = asd[incr]/ asd[0];
        ad[0] = asd[0] / (1 + r);
        out[0] = ((3*r+2) * qdu[0] + r*qdu[incr]) / ((1+r)*(1+r));
        // s'''(x(n-1)-) = s'''(x(n-1)+)
        r = asd[(sizeOfIn-3)*incr] / asd[(sizeOfIn-2)*incr];
        ad[(sizeOfIn-1)*incr] = asd[(sizeOfIn-2)*incr] / (1 + r);
        out[(sizeOfIn-1) * incr] = ((3*r+2) * qdu[(sizeOfIn-2)*incr] + r*qdu[(sizeOfIn-3)*incr]) / ((1+r)*(1+r));
        triDiagLDLSolve(ad, asd, out, sizeOfIn, incr);
    }
    else if(spType == CLAMPED)
    {
        // d(1) and d(n) are already known
        out[incr] = out[incr] - out[0] * asd[0];
        out[(sizeOfIn-2)*incr] = out[(sizeOfIn-2)*incr] - out[(sizeOfIn-1)*incr] * asd[(sizeOfIn-2)*incr];
        triDiagLDLSolve(ad+incr, asd+incr, out+incr, sizeOfIn-2, incr);
    }
    else if(spType == PERIODIC)
    {
        ad[0] = 2 * (asd[0] + asd[(sizeOfIn-2)*incr]);
        out[0] = 3 * (qdu[0] + qdu[(sizeOfIn-2)*incr]);

        qdu[0] = asd[(sizeOfIn-2)*incr];
        for(int i = incr; i < (sizeOfIn-3)*incr; i+=incr)
        {
            qdu[i] = 0;
        }

        qdu[(sizeOfIn-3)*incr] = asd[(sizeOfIn-3)*incr];

        cyclicTriDiagLDLtSolve(ad, asd, qdu, out, sizeOfIn-1, incr);
        out[(sizeOfIn-1) * incr] = out[0];
    }
}

__global__ void coef_bicubic_kernel(double* X, double* Y, double* Z,
                                    int sizeOfX, int sizeOfY,
                                    double* P, double* Q, double* R, double* C)
{
    int iPosY = blockIdx.x * blockDim.x + threadIdx.x;
    int iPosX = blockIdx.y * blockDim.y + threadIdx.y;
    if(iPosX > sizeOfX-2 || iPosY > sizeOfY-2)
    {
        return;
    }

    int iPos    = iPosY * sizeOfX + iPosX;
    int iPosCp1 = (iPosY+1) * sizeOfX + iPosX;
    int iPosRp1 = iPosY * sizeOfX + iPosX + 1;

    double dy = 1 / (Y[iPosY+1] - Y[iPosY]);
    double dx = 1 / (X[iPosX+1] - X[iPosX]);

    double a = 0;
    double b = 0;
    double c = 0;
    double d = 0;

    C[computeCPos(0,0,iPosX,iPosY,sizeOfX)] = Z[iPos];
    C[computeCPos(1,0,iPosX,iPosY,sizeOfX)] = P[iPos];
    C[computeCPos(0,1,iPosX,iPosY,sizeOfX)] = Q[iPos];
    C[computeCPos(1,1,iPosX,iPosY,sizeOfX)] = R[iPos];

    a = (Z[iPosRp1] - Z[iPos]) * dx;
    C[computeCPos(2,0,iPosX,iPosY,sizeOfX)] = (3*a - 2*P[iPos] - P[iPosRp1])*dx;
    C[computeCPos(3,0,iPosX,iPosY,sizeOfX)] = (P[iPosRp1] + P[iPos] - 2*a)*dx*dx;

    a = (Z[iPosCp1] - Z[iPos]) * dy;
    C[computeCPos(0,2,iPosX,iPosY,sizeOfX)] = (3*a - 2*Q[iPos] - Q[iPosCp1])*dy;
    C[computeCPos(0,3,iPosX,iPosY,sizeOfX)] = (Q[iPosCp1] + Q[iPos] - 2*a)*dy*dy;

    a = (Q[iPosRp1] - Q[iPos]) * dx;
    C[computeCPos(2,1,iPosX,iPosY,sizeOfX)] = (3*a - 2*R[iPos] - R[iPosRp1])*dx;
    C[computeCPos(3,1,iPosX,iPosY,sizeOfX)] = (R[iPosRp1] + R[iPos] - 2*a)*dx*dx;

    a = (P[iPosCp1] - P[iPos]) * dy;
    C[computeCPos(1,2,iPosX,iPosY,sizeOfX)] = (3*a - 2*R[iPos] - R[iPosCp1])*dy;
    C[computeCPos(1,3,iPosX,iPosY,sizeOfX)] = (R[iPosCp1] + R[iPos] - 2*a)*dy*dy;

    a = (Z[iPosCp1+1] + Z[iPos] - Z[iPosRp1] - Z[iPosCp1]) * dx * dx *dy * dy
        - (P[iPosCp1] - P[iPos]) * dx *dy *dy
        - (Q[iPosRp1] - Q[iPos]) * dx *dx *dy
        + R[iPos] * dx *dy;
    b = (P[iPosCp1+1] + P[iPos] - P[iPosRp1] - P[iPosCp1]) * dx * dy * dy
        - (R[iPosRp1] - R[iPos]) * dx * dy;
    c = (Q[iPosCp1+1] + Q[iPos] - Q[iPosRp1] - Q[iPosCp1]) * dx * dx * dy
        - (R[iPosCp1] - R[iPos]) * dx * dy;
    d = (R[iPosCp1+1] + R[iPos] - R[iPosRp1] - R[iPosCp1]) * dx * dy;

    C[computeCPos(2,2,iPosX,iPosY,sizeOfX)] = 9 * a - 3 * b - 3 * c + d;
    C[computeCPos(2,3,iPosX,iPosY,sizeOfX)] = (-6 * a + 2 * b + 3 * c - d) * dy;
    C[computeCPos(3,2,iPosX,iPosY,sizeOfX)] = (-6 * a + 3 * b + 2 * c - d) * dx;
    C[computeCPos(3,3,iPosX,iPosY,sizeOfX)] = (4 * a - 2 * b - 2 * c + d) * dx * dy;
}

/*******************************************/
/************ Bicubic Sub Splin ************/
/*******************************************/
cudaError_t cudaBicubicSubSplin(double* X, double* Y, double* Z, int sizeOfX, int sizeOfY,
                                double* P, double* Q, double* R, SplineType spType, double* C)
{
    cudaError_t cudaStat = cudaGetLastError();

    try
    {
        // get gpu properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        int threadMax = deviceProp.maxThreadsDim[0];
        int iDimBlock = 0;
        int iDimGrid = 0;

        if (spType == MONOTONE)// DPCHIM subroutine
        {
            // p = du/dx
            iDimBlock = (sizeOfY > threadMax) ? threadMax : sizeOfY;
            iDimGrid  = (int)ceil((float)sizeOfY/iDimBlock);
            dim3 colblock(1, iDimBlock, 1);
            dim3 colgrid(1, iDimGrid, 1);

            dpchim_kernel<<<colgrid, colblock>>>(sizeOfX, sizeOfY, X, Z, P, 1);

            // q = du/dy
            iDimBlock = (sizeOfX > threadMax) ? threadMax : sizeOfX;
            iDimGrid  = (int)ceil((float)sizeOfX/iDimBlock);
            dim3 rowblock(iDimBlock, 1, 1);
            dim3 rowgrid(iDimGrid, 1, 1);

            dpchim_kernel<<<rowgrid, rowblock>>>(sizeOfY, sizeOfX, Y, Z, Q, sizeOfX);
            cudaStat = cudaThreadSynchronize();
            if (cudaStat != cudaSuccess) throw cudaStat;

            // r = d2 u/ dx dy  approchee via  dq / dx
            dpchim_kernel<<<colgrid, colblock>>>(sizeOfX, sizeOfY, X, Q, R, 1);
        }
        else if (spType == FAST_PERIODIC)// DERIVD subroutine
        {// approximation des derivees partielles par methode simple
            // p = du/dx
            iDimBlock = (sizeOfY > threadMax) ? threadMax : sizeOfY;
            iDimGrid  = (int)ceil((float)sizeOfY/iDimBlock);
            dim3 colblock(1, iDimBlock, 1);
            dim3 colgrid(1, iDimGrid, 1);

            derivd_FP_kernel<<<colgrid, colblock>>>(sizeOfX, sizeOfY, X, Z, P, 1);

            // q = du/dy
            iDimBlock = (sizeOfX > threadMax) ? threadMax : sizeOfX;
            iDimGrid  = (int)ceil((float)sizeOfX/iDimBlock);
            dim3 rowblock(iDimBlock, 1, 1);
            dim3 rowgrid(iDimGrid, 1, 1);

            derivd_FP_kernel<<<rowgrid, rowblock>>>(sizeOfY, sizeOfX, Y, Z, Q, sizeOfX);
            cudaStat = cudaThreadSynchronize();
            if (cudaStat != cudaSuccess) throw cudaStat;

            // r = d2 u/ dx dy  approchee via  dq / dx
            derivd_FP_kernel<<<colgrid, colblock>>>(sizeOfX, sizeOfY, X, Q, R, 1);
        }
        else if (spType == FAST)// DERIVD subroutine
        {// approximation des derivees partielles par methode simple
            // p = du/dx
            iDimBlock = (sizeOfY > threadMax) ? threadMax : sizeOfY;
            iDimGrid  = (int)ceil((float)sizeOfY/iDimBlock);
            dim3 colblock(1, iDimBlock, 1);
            dim3 colgrid(1, iDimGrid, 1);

            derivd_F_kernel<<<colgrid, colblock>>>(sizeOfX, sizeOfY, X, Z, P, 1);

            // q = du/dy
            iDimBlock = (sizeOfX > threadMax) ? threadMax : sizeOfX;
            iDimGrid  = (int)ceil((float)sizeOfX/iDimBlock);
            dim3 rowblock(iDimBlock, 1, 1);
            dim3 rowgrid(iDimGrid, 1, 1);

            derivd_F_kernel<<<rowgrid, rowblock>>>(sizeOfY, sizeOfX, Y, Z, Q, sizeOfX);
            cudaStat = cudaThreadSynchronize();
            if (cudaStat != cudaSuccess) throw cudaStat;

            // r = d2 u/ dx dy  approchee via  dq / dx
            derivd_F_kernel<<<colgrid, colblock>>>(sizeOfX, sizeOfY, X, Q, R, 1);
        }

        cudaStat = cudaThreadSynchronize();
        if (cudaStat != cudaSuccess) throw cudaStat;

        // calculs des coefficients dans les bases (x-x(i))^k (y-y(j))^l  0<= k,l <= 3
        // pour evaluation rapide via Horner par la suite
        int iDimBlockX = (sizeOfX-1 > 16) ? 16 : sizeOfX-1;
        int iDimGridX  = (int)ceil((float)(sizeOfX-1)/iDimBlockX);
        int iDimBlockY = (sizeOfY-1 > 16) ? 16 : sizeOfY-1;
        int iDimGridY  = (int)ceil((float)(sizeOfY-1)/iDimBlockY);
        // sizeOfX = number of line of Z, y of block/dim is vertical axis.
        dim3 block(iDimBlockY, iDimBlockX, 1);
        dim3 grid(iDimGridY, iDimGridX, 1);

        coef_bicubic_kernel<<<grid, block>>>(X, Y, Z, sizeOfX, sizeOfY, P, Q, R, C);

        cudaStat = cudaThreadSynchronize();
        if (cudaStat != cudaSuccess) throw cudaStat;
        return cudaSuccess;
    }
    catch(cudaError_t cudaE)
    {
        return cudaE;
    }
}

/***************************************/
/************ Bicubic Splin ************/
/***************************************/
cudaError_t cudaBicubicSplin(double* X, double* Y, double* Z,
                             int sizeOfX, int sizeOfY,
                             double* P, double* Q, double* R,
                             double* Ad, double* Asd, double* Qdu,
                             SplineType spType, double* C)
{
    cudaError_t cudaStat = cudaGetLastError();

    try
    {
        // get gpu properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        cudaStat = cudaGetLastError();
        if (cudaStat != cudaSuccess) throw cudaStat;

        int threadMax = deviceProp.maxThreadsDim[0];
        int iDimBlock = 0;
        int iDimGrid  = 0;

        // compute du/dx
        iDimBlock = (sizeOfY > threadMax) ? threadMax : sizeOfY;
        iDimGrid  = (int)ceil((float)sizeOfY/iDimBlock);
        dim3 colblock(1, iDimBlock, 1);
        dim3 colgrid(1, iDimGrid, 1);

        if(sizeOfX == 3 && spType == NOT_A_KNOT)
        {
            derivd_F_kernel<<<colgrid, colblock>>>(sizeOfX, sizeOfY, X, Z, P, 1);
        }
        else
        {
            splinCub_kernel<<<colgrid, colblock>>>(sizeOfX, sizeOfY, X, Z, P,
                                                   Ad, Asd, Qdu, 1, spType);
        }

        // compute du/dy
        iDimBlock = (sizeOfX > threadMax) ? threadMax : sizeOfX;
        iDimGrid  = (int)ceil((float)sizeOfX/iDimBlock);
        dim3 rowblock(iDimBlock, 1, 1);
        dim3 rowgrid(iDimGrid, 1, 1);
        if(sizeOfY == 3 && spType == NOT_A_KNOT)
        {
            derivd_F_kernel<<<rowgrid, rowblock>>>(sizeOfY, sizeOfX, Y, Z, Q, sizeOfX);
        }
        else
        {
            splinCub_kernel<<<rowgrid, rowblock>>>(sizeOfY, sizeOfX, Y, Z, Q,
                                                   Ad, Asd, Qdu, sizeOfX, spType);
        }

        cudaStat = cudaThreadSynchronize();
        if (cudaStat != cudaSuccess) throw cudaStat;

        // compute ddu/dxdy
        dim3 oneblock(1, 1, 1);
        dim3 onegrid(1, 1, 1);
        int iPos = (sizeOfY-1)*sizeOfX;
        if(sizeOfX == 3 && spType == NOT_A_KNOT)
        {
            derivd_F_kernel<<<onegrid, oneblock>>>(sizeOfX, 1, X, Q, R, 1);
            derivd_F_kernel<<<onegrid, oneblock>>>(sizeOfX, 1, X, Q+iPos, R+iPos, 1);
        }
        else
        {
            splinCub_kernel<<<onegrid, oneblock>>>(sizeOfX, 1, X, Q, R,
                                                   Ad, Asd, Qdu, 1, spType);
            splinCub_kernel<<<onegrid, oneblock>>>(sizeOfX, 1, X, Q+iPos, R+iPos,
                                                   Ad, Asd, Qdu, 1, spType);
        }

        if(sizeOfY == 3 && spType == NOT_A_KNOT)
        {
            derivd_F_kernel<<<rowgrid, rowblock>>>(sizeOfY, sizeOfX, Y, P, R, sizeOfX);
        }
        else
        {
            splinCub_kernel<<<rowgrid, rowblock>>>(sizeOfY, sizeOfX, Y, P, R,
                                                   Ad, Asd, Qdu, sizeOfX, CLAMPED);
        }

        cudaStat = cudaThreadSynchronize();
        if (cudaStat != cudaSuccess) throw cudaStat;

        // calculs des coefficients dans les bases (x-x(i))^k (y-y(j))^l  0<= k,l <= 3
        // pour evaluation rapide via Horner par la suite
        int iDimBlockX = (sizeOfX-1 > 16) ? 16 : sizeOfX-1;
        int iDimGridX  = (int)ceil((float)(sizeOfX-1)/iDimBlockX);
        int iDimBlockY = (sizeOfY-1 > 16) ? 16 : sizeOfY-1;
        int iDimGridY  = (int)ceil((float)(sizeOfY-1)/iDimBlockY);
        // sizeOfX = number of line of Z, y of block/dim is vertical axis.
        dim3 block(iDimBlockY, iDimBlockX, 1);
        dim3 grid(iDimGridY, iDimGridX, 1);

        coef_bicubic_kernel<<<grid, block>>>(X, Y, Z, sizeOfX, sizeOfY, P, Q, R, C);

        cudaStat = cudaThreadSynchronize();
        if (cudaStat != cudaSuccess) throw cudaStat;
        return cudaSuccess;
    }
    catch(cudaError_t cudaE)
    {
        return cudaE;
    }
}
