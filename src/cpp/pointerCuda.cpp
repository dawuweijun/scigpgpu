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

#include "pointerCuda.hxx"
#include "makecucomplex.h"
#include "cublas.h"
#include "cufft.h"
#include "idmax.h"
#include "idmin.h"
#include "dsum.h"
#include "zsum.h"
#include "matrixTranspose.h"
#include "extract.h"
#include "insert.h"
#include "initCudaMatrix.h"

PointerCuda::PointerCuda()
{
}

PointerCuda::PointerCuda(int rows, int cols, bool complex)
{
    if (complex)
    {
        _ptr = getCudaContext()->genMatrix<double>(getCudaQueue(), rows * cols * 2);
    }
    else
    {
        _ptr = getCudaContext()->genMatrix<double>(getCudaQueue(), rows * cols);
    }

    _bComplex   = complex;
    _iCols      = cols;
    _iRows      = rows;

    _iDims      = 2;
    _iDimsArray = new int[2];
    _iDimsArray[0] = rows;
    _iDimsArray[1] = cols;
    _iTotalSize = rows * cols;
}

PointerCuda::PointerCuda(int dims, int* dimsArray, bool complex)
{
    int iTotalSize = 1;
    for ( int i = 0; i < dims; i++)
    {
        iTotalSize *= dimsArray[i];
    }

    if (complex)
    {
        _ptr = getCudaContext()->genMatrix<double>(getCudaQueue(), iTotalSize * 2);
    }
    else
    {
        _ptr = getCudaContext()->genMatrix<double>(getCudaQueue(), iTotalSize);
    }

    _bComplex    = complex;
    _iRows      = dimsArray[0];
    _iCols      = 1;

    if (dims > 1)
    {
        _iCols = dimsArray[1];
    }

    _iDims = dims;
    _iDimsArray = dimsArray;
    _iTotalSize = iTotalSize;
}

PointerCuda::PointerCuda(double* data, int rows, int cols)
{
    _ptr = getCudaContext()->genMatrix<double>(getCudaQueue(), rows * cols, data);
    _bComplex   = false;
    _iCols      = cols;
    _iRows      = rows;

    _iDims      = 2;
    _iDimsArray = new int[2];
    _iDimsArray[0] = rows;
    _iDimsArray[1] = cols;
    _iTotalSize = rows * cols;
}

PointerCuda::PointerCuda(double* dataReal, double* dataImg, int rows, int cols)
{
    double* d = NULL;
    cublasAlloc(rows * cols, sizeof(cuDoubleComplex), (void**)&d);
    _cudaStat = writecucomplex(dataReal, dataImg, rows, cols, (cuDoubleComplex *)d);
    if (_cudaStat != cudaSuccess)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
    }
    _ptr = getCudaContext()->genMatrix<double>(getCudaQueue(), rows * cols);
    _ptr->set_ptr(d);
    _bComplex   = true;
    _iCols      = cols;
    _iRows      = rows;
    _iDims      = 2;
    _iDimsArray = new int[2];
    _iDimsArray[0] = rows;
    _iDimsArray[1] = cols;
    _iTotalSize = rows * cols;
}

PointerCuda::PointerCuda(double* data, int dims, int* dimsArray)
{
    int iTotalSize = 1;
    for ( int i = 0; i < dims; i++)
    {
        iTotalSize *= dimsArray[i];
    }

    _ptr = getCudaContext()->genMatrix<double>(getCudaQueue(), iTotalSize, data);
    _bComplex   = false;
    _iRows      = dimsArray[0];
    _iCols      = 1;

    if (dims > 1)
    {
        _iCols = dimsArray[1];
    }

    _iDims = dims;
    _iDimsArray = dimsArray;
    _iTotalSize = iTotalSize;
}

PointerCuda::PointerCuda(double* dataReal, double* dataImg, int dims, int* dimsArray)
{
    double* d = NULL;
    int iTotalSize = 1;

    for ( int i = 0; i < dims; i++)
    {
        iTotalSize *= dimsArray[i];
    }

    cublasAlloc(iTotalSize, sizeof(cuDoubleComplex), (void**)&d);
    _cudaStat = writecucomplex(dataReal, dataImg, 1, iTotalSize, (cuDoubleComplex *)d);

    if (_cudaStat != cudaSuccess)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
    }

    _ptr = getCudaContext()->genMatrix<double>(getCudaQueue(), iTotalSize);
    _ptr->set_ptr(d);
    _bComplex    = true;
    _iRows      = dimsArray[0];
    _iCols      = 1;

    if (dims > 1)
    {
        _iCols = dimsArray[1];
    }

    _iDims = dims;
    _iDimsArray = dimsArray;
    _iTotalSize = iTotalSize;
}

GpuPointer* PointerCuda::operator+(GpuPointer &b)
{
    PointerCuda* result = NULL;

    bool bComplex_A = _bComplex;
    bool bComplex_B = b.isGpuComplex();
    int na          = getSize();
    int nb          = b.getSize();
    double* d_A     = getGpuPtr();
    double* d_B     = b.getGpuPtr();

    double alpha        = 1.0f;

    cuDoubleComplex alphaComplex;
    alphaComplex.x      = 1.0f;
    alphaComplex.y      = 0.0f;

    cublasStatus status;

    bool complex = false;
    if (_bComplex || bComplex_B)
    {
        complex = true;
    }

    // Allocate result pointer
    if (na == 1)
    {
        result = new PointerCuda(b.getRows(), b.getCols(), complex);
    }
    else
    {
        result = new PointerCuda(_iRows, _iCols, complex);
    }

    double* d_C = result->getGpuPtr();

    if (!bComplex_A && !bComplex_B)
    {
        if (nb == 1 && na > 1)
        {
            cublasDcopy (na, d_A, 1, d_C, 1);
            status = cublasGetError();
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
            }
            cublasDaxpy(na, alpha, d_B, 0, d_C, 1);
        }
        else if (na == 1 && nb > 1)
        {
            cublasDcopy (nb, d_B, 1, d_C, 1);
            status = cublasGetError();
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
            }
            cublasDaxpy(nb, alpha, d_A, 0, d_C, 1);
        }
        else
        {
            cublasDcopy (nb, d_B, 1, d_C, 1);
            status = cublasGetError();
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
            }
            cublasDaxpy(nb, alpha, d_A, 1, d_C, 1);
        }
    }
    else
    {
        if (bComplex_B && !bComplex_A)
        {
            cuDoubleComplex* d_AComplex;
            status = cublasAlloc(na, sizeof(cuDoubleComplex), (void**)&d_AComplex);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
            }

            if (nb == 1 && na > 1)
            {
                _cudaStat = rewritecucomplex(d_A, _iRows, _iCols, d_AComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZcopy (na, d_AComplex, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(na, alphaComplex, (cuDoubleComplex*)d_B, 0, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }
            else if (na == 1 && nb > 1)
            {
                _cudaStat = rewritecucomplex(d_A, _iRows, _iCols, d_AComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZcopy (nb, (cuDoubleComplex*)d_B, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(nb, alphaComplex, d_AComplex, 0, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }
            else
            {
                _cudaStat = rewritecucomplex(d_A, _iRows, _iCols, d_AComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZcopy (nb, (cuDoubleComplex*)d_B, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(na, alphaComplex, d_AComplex, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }

            status = cublasFree(d_AComplex);
        }
        else if (!bComplex_B && bComplex_A)
        {
            cuDoubleComplex* d_BComplex;
            status = cublasAlloc(nb, sizeof(cuDoubleComplex), (void**)&d_BComplex);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
            }

            if (nb == 1 && na > 1)
            {
                _cudaStat = rewritecucomplex(d_B, b.getRows(), b.getCols(), d_BComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZcopy (na, (cuDoubleComplex*)d_A, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(na, alphaComplex, d_BComplex, 0, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }
            else if (na == 1 && nb > 1)
            {
                _cudaStat = rewritecucomplex(d_B, b.getRows(), b.getCols(), d_BComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZcopy (nb, d_BComplex, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(nb, alphaComplex, (cuDoubleComplex*)d_A, 0, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }
            else
            {
                _cudaStat = rewritecucomplex(d_B, b.getRows(), b.getCols(), d_BComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZcopy (nb, d_BComplex, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(na, alphaComplex, (cuDoubleComplex*)d_A, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }

            status = cublasFree(d_BComplex);
        }
        else
        {
            if (nb == 1 && na > 1)
            {
                cublasZcopy (na, (cuDoubleComplex*)d_A, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(na, alphaComplex, (cuDoubleComplex*)d_B, 0, (cuDoubleComplex*)d_C, 1);
            }
            else if (na == 1 && nb > 1)
            {
                cublasZcopy (nb, (cuDoubleComplex*)d_B, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(nb, alphaComplex, (cuDoubleComplex*)d_A, 0, (cuDoubleComplex*)d_C, 1);
            }
            else
            {
                cublasZcopy (nb, (cuDoubleComplex*)d_B, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(na, alphaComplex, (cuDoubleComplex*)d_A, 1, (cuDoubleComplex*)d_C, 1);
            }
        }
    }
    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
    }

    cudaThreadSynchronize();

    return result;
}

GpuPointer* PointerCuda::operator-(GpuPointer &b)
{
    PointerCuda* result = NULL;

    bool bComplex_A = _bComplex;
    bool bComplex_B = b.isGpuComplex();
    int na          = getSize();
    int nb          = b.getSize();
    double* d_A     = getGpuPtr();
    double* d_B     = b.getGpuPtr();

    double alpha    = 1.0f;

    cuDoubleComplex alphaComplex;
    alphaComplex.x = 1.0f;
    alphaComplex.y = 0.0f;
    cuDoubleComplex m1;
    m1.x = -1.0f;
    m1.y = 0.0f;

    cublasStatus status;

    bool complex = false;
    if (_bComplex || bComplex_B)
    {
        complex = true;
    }

    // Allocate result pointer
    if (na == 1)
    {
        result = new PointerCuda(b.getRows(), b.getCols(), complex);
    }
    else
    {
        result = new PointerCuda(_iRows, _iCols, complex);
    }

    double* d_C = result->getGpuPtr();

    if (!bComplex_A && !bComplex_B)
    {
        if (nb == 1 && na > 1)
        {
            cublasDcopy (na, d_A, 1, d_C, 1);
            status = cublasGetError();
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
            }
            cublasDaxpy(na, -alpha, d_B, 0, d_C, 1);
        }
        else if (na == 1 && nb > 1)
        {
            cublasDcopy (nb, d_B, 1, d_C, 1);
            status = cublasGetError();
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
            }

            cublasDscal(nb, -1, d_C, 1);
            status = cublasGetError();
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
            }

            cublasDaxpy(nb, alpha, d_A, 0, d_C, 1);
        }
        else
        {
            cublasDcopy (nb, d_A, 1, d_C, 1);
            status = cublasGetError();
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
            }
            cublasDaxpy(nb, -alpha, d_B, 1, d_C, 1);
        }
    }
    else
    {
        if (bComplex_B && !bComplex_A)
        {
            cuDoubleComplex* d_AComplex;
            status = cublasAlloc(na, sizeof(cuDoubleComplex), (void**)&d_AComplex);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
            }

            if (nb == 1 && na > 1)
            {
                _cudaStat = rewritecucomplex(d_A, _iRows, _iCols, d_AComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZcopy (na, d_AComplex, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(na, m1, (cuDoubleComplex*)d_B, 0, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }
            else if (na == 1 && nb > 1)
            {
                _cudaStat = rewritecucomplex(d_A, _iRows, _iCols, d_AComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZcopy (nb, (cuDoubleComplex*)d_B, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZscal(nb, m1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(nb, alphaComplex, d_AComplex, 0, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }
            else
            {
                _cudaStat = rewritecucomplex(d_A, _iRows, _iCols, d_AComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZcopy (nb, d_AComplex, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(na, m1, (cuDoubleComplex*)d_B, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }

            status = cublasFree(d_AComplex);
        }
        else if (!bComplex_B && bComplex_A)
        {
            cuDoubleComplex* d_BComplex;
            status = cublasAlloc(nb, sizeof(cuDoubleComplex), (void**)&d_BComplex);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
            }

            if (nb == 1 && na > 1)
            {
                _cudaStat = rewritecucomplex(d_B, b.getRows(), b.getCols(), d_BComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZcopy (na, (cuDoubleComplex*)d_A, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(na, m1, d_BComplex, 0, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }
            else if (na == 1 && nb > 1)
            {
                _cudaStat = rewritecucomplex(d_B, b.getRows(), b.getCols(), d_BComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZcopy (nb, d_BComplex, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZscal(nb, m1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(nb, alphaComplex, (cuDoubleComplex*)d_A, 0, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }
            else
            {
                _cudaStat = rewritecucomplex(d_B, b.getRows(), b.getCols(), d_BComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZcopy (nb, (cuDoubleComplex*)d_A, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(na, m1, d_BComplex, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }

            status = cublasFree(d_BComplex);
        }
        else
        {
            if (nb == 1 && na > 1)
            {
                cublasZcopy (na, (cuDoubleComplex*)d_A, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(na, m1, (cuDoubleComplex*)d_B, 0, (cuDoubleComplex*)d_C, 1);
            }
            else if (na == 1 && nb > 1)
            {
                cublasZcopy (nb, (cuDoubleComplex*)d_B, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZscal(nb, m1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(nb, alphaComplex, (cuDoubleComplex*)d_A, 0, (cuDoubleComplex*)d_C, 1);
            }
            else
            {
                cublasZcopy (nb, (cuDoubleComplex*)d_A, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZaxpy(na, m1, (cuDoubleComplex*)d_B, 1, (cuDoubleComplex*)d_C, 1);
            }
        }
    }

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
    }

    cudaThreadSynchronize();

    return result;
}

GpuPointer* PointerCuda::operator*(GpuPointer &b)
{
    PointerCuda* result = NULL;

    bool bComplex_A = _bComplex;
    bool bComplex_B = b.isGpuComplex();
    int na          = getSize();
    int nb          = b.getSize();
    double* d_A     = getGpuPtr();
    double* d_B     = b.getGpuPtr();

    double alpha    = 1.0f;
    double beta     = 0.0f;

    cuDoubleComplex alphaComplex;
    alphaComplex.x      = 1.0f;
    alphaComplex.y      = 0.0f;
    cuDoubleComplex betaComplex;
    betaComplex.x       = 0.0f;
    betaComplex.y       = 0.0f;

    cublasStatus status;

    bool complex = false;
    if (_bComplex || bComplex_B)
    {
        complex = true;
    }

    // Allocate result pointer
    if (na == 1)
    {
        result = new PointerCuda(b.getRows(), b.getCols(), complex);
    }
    else if (nb == 1)
    {
        result = new PointerCuda(_iRows, _iCols, complex);
    }
    else
    {
        result = new PointerCuda(_iRows, b.getCols(), complex);
    }

    double* d_C = result->getGpuPtr();

    if (!bComplex_A && !bComplex_B)
    {
        if (nb == 1 && na > 1)
        {
            double h_B = 0;
            b.getData(&h_B);
            cublasDcopy (na, d_A, 1, d_C, 1);
            status = cublasGetError();
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
            }
            cublasDscal(na, h_B, d_C, 1);
        }
        else if (na == 1 && nb > 1)
        {
            double h_A = 0;
            getData(&h_A);
            cublasDcopy (nb, d_B, 1, d_C, 1);
            status = cublasGetError();
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
            }
            cublasDscal(nb, h_A, d_C, 1);
        }
        else
        {
            cublasDgemm('n', 'n', _iRows, b.getCols(), _iCols /* or b.getRows()*/, alpha, d_A, _iRows, d_B, b.getRows(), beta, d_C, _iRows);
        }
    }
    else
    {
        if (na == 1 && nb == 1)
        {
            if (bComplex_A)
            {
                if (!bComplex_B)
                {
                    double h_B = 0;
                    b.getData(&h_B);
                    cublasZcopy (na, (cuDoubleComplex*)d_A, 1, (cuDoubleComplex*)d_C, 1);
                    status = cublasGetError();
                    if (status != CUBLAS_STATUS_SUCCESS)
                    {
                        GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                    }
                    cublasZdscal(na, h_B, (cuDoubleComplex*)d_C, 1);
                }
                else
                {
                    cublasZgemm('n', 'n', _iRows, b.getCols(), _iCols, alphaComplex, (cuDoubleComplex *)d_A, _iRows, (cuDoubleComplex *)d_B, b.getRows(), betaComplex, (cuDoubleComplex *)d_C, _iRows);
                }
            }
            else // bComplex_B
            {
                double h_A = 0;
                getData(&h_A);
                cublasZcopy (nb, (cuDoubleComplex*)d_B, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZdscal(nb, h_A, (cuDoubleComplex*)d_C, 1);
            }
        }
        else if (nb == 1)
        {
            if (bComplex_B)
            {
                if (!bComplex_A)
                {
                    cuDoubleComplex* d_AComplex;
                    status = cublasAlloc(na, sizeof(cuDoubleComplex), (void**)&d_AComplex);
                    if (status != CUBLAS_STATUS_SUCCESS)
                    {
                        GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                    }
                    _cudaStat = rewritecucomplex(d_A, _iRows, _iCols, d_AComplex);
                    if (_cudaStat != cudaSuccess)
                    {
                        GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                    }
                    cublasZcopy (na, d_AComplex, 1, (cuDoubleComplex*)d_C, 1);
                    status = cublasFree(d_AComplex);
                    if (status != CUBLAS_STATUS_SUCCESS)
                    {
                        GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                    }
                }
                else
                {
                    cublasZcopy (na, (cuDoubleComplex*)d_A, 1, (cuDoubleComplex*)d_C, 1);
                    status = cublasGetError();
                    if (status != CUBLAS_STATUS_SUCCESS)
                    {
                        GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                    }
                }
                cuDoubleComplex h_B;
                b.getData(&(h_B.x), &(h_B.y));
                cublasZscal(na, h_B, (cuDoubleComplex*)d_C, 1); // cuDoubleComplex* x cuDoubleComplex
            }
            else
            {
                double h_B = 0;
                b.getData(&h_B);
                cublasZcopy (na, (cuDoubleComplex*)d_A, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZdscal(na, h_B, (cuDoubleComplex*)d_C, 1);    //  cuDoubleComplex* x double
            }
        }
        else if (na == 1)
        {
            if (bComplex_A)
            {
                if (!bComplex_B)
                {
                    cuDoubleComplex* d_BComplex;
                    status = cublasAlloc(nb, sizeof(cuDoubleComplex), (void**)&d_BComplex);
                    if (status != CUBLAS_STATUS_SUCCESS)
                    {
                        GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                    }
                    _cudaStat = rewritecucomplex(d_B, b.getRows(), b.getCols(), d_BComplex);
                    if (_cudaStat != cudaSuccess)
                    {
                        GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                    }
                    cublasZcopy (nb, d_BComplex, 1, (cuDoubleComplex*)d_C, 1);
                    status = cublasGetError();
                    if (status != CUBLAS_STATUS_SUCCESS)
                    {
                        GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                    }
                    status = cublasFree(d_BComplex);
                    if (status != CUBLAS_STATUS_SUCCESS)
                    {
                        GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                    }
                }
                else
                {
                    cublasZcopy (nb, (cuDoubleComplex*)d_B, 1, (cuDoubleComplex*)d_C, 1);
                    status = cublasGetError();
                    if (status != CUBLAS_STATUS_SUCCESS)
                    {
                        GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                    }
                }
                cuDoubleComplex h_A;
                getData(&(h_A.x), &(h_A.y));
                cublasZscal(nb, h_A, (cuDoubleComplex*)d_C, 1); // cuDoubleComplex x cuDoubleComplex*
            }
            else
            {
                double h_A = 0;
                getData(&h_A);
                cublasZcopy (nb, (cuDoubleComplex*)d_B, 1, (cuDoubleComplex*)d_C, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                cublasZdscal(nb, h_A, (cuDoubleComplex*)d_C, 1);    // double x cuDoubleComplex*
            }
        }
        else
        {
            if (!bComplex_A)
            {
                cuDoubleComplex* d_AComplex;
                status = cublasAlloc(na, sizeof(cuDoubleComplex), (void**)&d_AComplex);
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                _cudaStat = rewritecucomplex(d_A, _iRows, _iCols, d_AComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZgemm('n', 'n', _iRows, b.getCols(), _iCols, alphaComplex, d_AComplex, _iRows, (cuDoubleComplex *)d_B, b.getRows(), betaComplex, (cuDoubleComplex *)d_C, _iRows);
                status = cublasFree(d_AComplex);
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }
            else if (!bComplex_B)
            {
                cuDoubleComplex* d_BComplex;
                status = cublasAlloc(nb, sizeof(cuDoubleComplex), (void**)&d_BComplex);
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
                _cudaStat = rewritecucomplex(d_B, b.getRows(), b.getCols(), d_BComplex);
                if (_cudaStat != cudaSuccess)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
                }
                cublasZgemm('n', 'n', _iRows, b.getCols(), _iCols, alphaComplex, (cuDoubleComplex *)d_A, _iRows, d_BComplex, b.getRows(), betaComplex, (cuDoubleComplex *)d_C, _iRows);
                status = cublasFree(d_BComplex);
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
                }
            }
            else
            {
                cublasZgemm('n', 'n', _iRows, b.getCols(), _iCols, alphaComplex, (cuDoubleComplex *)d_A, _iRows, (cuDoubleComplex *)d_B, b.getRows(), betaComplex, (cuDoubleComplex *)d_C, _iRows);
            }
        }
    }

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
    }

    cudaThreadSynchronize();

    return result;
}

GpuPointer* PointerCuda::transpose()
{
    PointerCuda* result = new PointerCuda(_iCols, _iRows, _bComplex);

    if (!_bComplex)
    {
        _cudaStat = cudaTranspose(getGpuPtr(), result->getGpuPtr(), _iRows, _iCols);
    }
    else
    {
        _cudaStat = cudaZTranspose((cuDoubleComplex*)getGpuPtr(), (cuDoubleComplex*)result->getGpuPtr(), _iRows, _iCols);
    }

    if (_cudaStat != cudaSuccess)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
    }

    cudaThreadSynchronize();

    return result;
}

GpuPointer* PointerCuda::clone()
{
    cublasStatus status;
    PointerCuda* clone = new PointerCuda(_iDims, _iDimsArray, _bComplex);

    if (_bComplex)
    {
        cublasZcopy(_iTotalSize, (cuDoubleComplex*)getGpuPtr(), 1, (cuDoubleComplex*)clone->getGpuPtr(), 1);
    }
    else
    {
        cublasDcopy(_iTotalSize, getGpuPtr(), 1, clone->getGpuPtr(), 1);
    }

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
    }

    cudaThreadSynchronize();

    return clone;
}
double PointerCuda::getSum()
{
    int alpha = 1;
    double res = 0;
    cublasStatus status;

    // Performs operation
    if (!_bComplex)
    {
        //res = cublasDasum(na,(double*)d_A,1);
        _cudaStat = cudaDsum(getSize(), getGpuPtr(), &res);
        if (_cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
        }
    }

    cudaThreadSynchronize();

    return res;
}

cuDoubleComplex PointerCuda::getComplexSum()
{
    int alpha = 1;
    cuDoubleComplex resComplex;
    cublasStatus status;

    // Performs operation
    if (!_bComplex)
    {
        resComplex.x = getSum();
        resComplex.y = 0;
    }
    else
    {
        //res = cublasDzasum(na,(cuDoubleComplex *)d_A,1);
        _cudaStat = cudaZsum(getSize(), (cuDoubleComplex*)getGpuPtr(), &resComplex);
        if (_cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
        }

    }

    cudaThreadSynchronize();

    return resComplex;
}

double PointerCuda::getNorm()
{
    cublasStatus status;
    int alpha = 1;
    double res;

    // Performs operation using cublas
    if (!_bComplex)
    {
        res = cublasDnrm2(getSize(), getGpuPtr(), alpha);
    }
    else
    {
        res = cublasDznrm2(getSize(), (cuDoubleComplex*)getGpuPtr(), alpha);
    }

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
    }

    cudaThreadSynchronize();

    return res;

}
double PointerCuda::getMin()
{
    double res = 0;
    if (!_bComplex)
    {
        _cudaStat = cudaIdmin(getSize(), getGpuPtr(), &res);
        if (_cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
        }
    }

    cudaThreadSynchronize();

    return res;
}

cuDoubleComplex PointerCuda::getComplexMin()
{
    int alpha = 1;
    cuDoubleComplex res;
    cublasStatus status;

    if (!_bComplex)
    {
        res.x = getMin();
        res.y = 0;
    }
    else
    {
        int pos = cublasIzamin(getSize(), (cuDoubleComplex*)_ptr->get_ptr(), alpha);
        status = cublasGetError();
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
        }

        pos--; // pos begin at 1 but pointer begin at 0

        _cudaStat = cudaMemcpy((void**)&res, getGpuPtr() + (pos * 2), sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost); // pos is the position in the matrix of cuDoubleComplex and one cuDoubleComplex is composed of two double.
        if (_cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
        }
    }

    cudaThreadSynchronize();

    return res;
}

double PointerCuda::getMax()
{
    double res = 0;
    if (!_bComplex)
    {
        _cudaStat = cudaIdmax(getSize(), getGpuPtr(), &res);
        if (_cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
        }
    }

    cudaThreadSynchronize();

    return res;
}

cuDoubleComplex PointerCuda::getComplexMax()
{
    int alpha = 1;
    cuDoubleComplex res;
    cublasStatus status;

    if (!_bComplex)
    {
        res.x = getMax();
        res.y = 0;
    }
    else
    {
        int pos = cublasIzamax(getSize(), (cuDoubleComplex*)_ptr->get_ptr(), alpha);
        status = cublasGetError();
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
        }

        pos--; // pos begin at 1 but pointer begin at 0

        // pos is the position in the matrix of cuDoubleComplex and one cuDoubleComplex is composed of two double.
        _cudaStat = cudaMemcpy((void**)&res, getGpuPtr() + (pos * 2), sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        if (_cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
        }
    }

    cudaThreadSynchronize();

    return res;
}

GpuPointer* PointerCuda::FFT(int iSign, int* piDim, int iSizeDim, int* piIncr)
{
    cufftHandle plan;
    cufftResult cuffterror;
    cufftType fftType = CUFFT_Z2Z;
    /************************************************************************************/
    /*  For real‐to‐complex FFTs (CUFFT_D2Z),                                           */
    /*  the output array holds only the non‐redundant complex coefficients.             */
    /*  So we transform the real input data to complex (set the imaginary part to 0)    */
    /*  and perform complex-to‐complex FFTs (CUFFT_Z2Z).                                */
    /************************************************************************************/

    PointerCuda* result = new PointerCuda(_iDims, _iDimsArray, true);
    int iNewSize = 1;
    int iNoDims  = 0;
    int iDist    = 0;
    int iIncr    = 1;
    bool bDelete = false;

    if (piDim == NULL)
    {
        iSizeDim    = _iDims;
        piDim       = new int[_iDims];
        bDelete     = true;

        // The dims in cufft are inverted.
        for (int i = 0; i < _iDims; i++)
        {
            piDim[i] = _iDimsArray[iSizeDim - i - 1];
        }
    }
    else
    {
        // The dims in cufft are inverted.
        iDist = 1;
        //        iIncr = piIncr[0];

        int* piTempDim = piDim;
        piDim = new int[iSizeDim];

        //        int* piTempIncr = piIncr;
        //        piIncr = new int[iSizeDim];

        for (int i = 0; i < iSizeDim; i++)
        {
            piDim[i]  = piTempDim[iSizeDim - i - 1];
            //piIncr[i] = piTempIncr[iSizeDim - i - 1];
        }

        delete piTempDim;
        //        delete piTempIncr;
    }

    for (int i = 0; i < iSizeDim; i++)
    {
        iNewSize *= piDim[i];
        if (iNewSize != 1 && piDim[i] == 1)
        {
            iNoDims++;
        }
    }

    // [1 n] => 2 dims
    // [n 1] => 1 dim
    // Create plan
    cuffterror = cufftPlanMany(&plan, iSizeDim - iNoDims, piDim, piIncr, iIncr, iDist, piIncr, iIncr, iDist, fftType, 1);
    if (cuffterror > 0)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)cuffterror, 2);
    }

    // exec fft
    //iSign = CUFFT_FORWARD -1 or CUFFT_INVERSE 1
    if (_bComplex)
    {
        cuffterror = cufftExecZ2Z(plan, (cuDoubleComplex*)getGpuPtr(), (cuDoubleComplex*)result->getGpuPtr(), iSign);
    }
    else
    {
        _cudaStat = rewritecucomplex(getGpuPtr(), _iRows, _iCols, (cuDoubleComplex *)result->getGpuPtr());
        if (_cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
        }

        cuffterror = cufftExecZ2Z(plan, (cuDoubleComplex*)result->getGpuPtr(), (cuDoubleComplex*)result->getGpuPtr(), iSign);
    }

    if (cuffterror > 0)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)cuffterror, 2);
    }

    if (iSign == 1) // CUFFT_INVERSE
    {
        double dScalBy = 1 / static_cast<double>(iNewSize);
        cublasZdscal(iNewSize, dScalBy, (cuDoubleComplex*)result->getGpuPtr(), 1);
        cublasStatus status = cublasGetError();
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
        }
    }

    // destroy plan
    cufftDestroy(plan);

    cudaThreadSynchronize();

    if (bDelete)
    {
        delete piDim;
        piDim = NULL;
    }

    return result;
}

int PointerCuda::getRows(void)
{
    return _iRows;
}
int PointerCuda::getCols(void)
{
    return _iCols;
}
void PointerCuda::setRows(int newRows)
{
    _iRows = newRows;
    _iDimsArray[0] = newRows;
    _iTotalSize = 1;
    for (int i = 0; i < _iDims; i++)
    {
        _iTotalSize *= _iDimsArray[i];
    }
}
void PointerCuda::setCols(int newCols)
{
    _iCols = newCols;
    _iDimsArray[1] = newCols;
    _iTotalSize = 1;
    for (int i = 0; i < _iDims; i++)
    {
        _iTotalSize *= _iDimsArray[i];
    }
}
int PointerCuda::getDims(void)
{
    return _iDims;
}
int* PointerCuda::getDimsArray(void)
{
    return _iDimsArray;
}
int PointerCuda::getSize(void)
{
    return _iTotalSize;
}
int PointerCuda::getSizeOfElem(void)
{
    int size = 0;
    if (_bComplex)
    {
        size = sizeof(cuDoubleComplex);
    }
    else
    {
        size = sizeof(double);
    }

    return size;
}
bool PointerCuda::isGpuComplex(void)
{
    return _bComplex;
}
GpuPointer::GpuType PointerCuda::getGpuType(void)
{
    return CudaType;
}
double* PointerCuda::getGpuPtr(void)
{
    return (double*)_ptr->get_ptr();
}
void PointerCuda::getData(double* h)
{
    _ptr->to_cpu_ptr(h);
}
void PointerCuda::getData(double* real, double* img)
{
    _cudaStat = readcucomplex(real, img, _iRows, _iCols, (cuDoubleComplex *)_ptr->get_ptr());
    if (_cudaStat != cudaSuccess)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
    }
}

void PointerCuda::initMatrix(double real)
{
    _cudaStat = initCudaMatrix(real, _iTotalSize, getGpuPtr());

    if (_cudaStat != cudaSuccess)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
    }
}

GpuPointer* PointerCuda::extract(GpuPointer* gpuPos)
{
    PointerCuda* ptrOut = new PointerCuda(gpuPos->getRows(), gpuPos->getCols(), isGpuComplex());
    int iErr = 0;
    if (isGpuComplex() == false)
    {
        _cudaStat = cudaExtract(getGpuPtr(), getSize(), ptrOut->getGpuPtr(), gpuPos->getGpuPtr(), gpuPos->getSize(), &iErr);
    }
    else
    {
        _cudaStat = cudaZExtract((cuDoubleComplex*)getGpuPtr(), getSize(), (cuDoubleComplex*)ptrOut->getGpuPtr(),
                                 gpuPos->getGpuPtr(), gpuPos->getSize(), &iErr);
    }

    if (_cudaStat != cudaSuccess)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
    }

    if (iErr)
    {
        delete ptrOut;
        ptrOut = NULL;
    }

    return ptrOut;
}
int PointerCuda::insert(GpuPointer* gpuData, GpuPointer* gpuPos, int isScalar)
{
    int iErr = 0;
    if (isGpuComplex() == false && gpuData->isGpuComplex() == false)
    {
        _cudaStat = cudaInsert(getGpuPtr(), getSize(), gpuData->getGpuPtr(), gpuPos->getGpuPtr(), gpuPos->getSize(), isScalar, &iErr);
    }
    else if (gpuData->isGpuComplex() && isGpuComplex() == false)
    {
        double* ptrCplx = NULL;
        cublasStatus status = cublasAlloc(getSize(), sizeof(cuDoubleComplex), (void**)&ptrCplx);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)status, 1);
        }

        _cudaStat = rewritecucomplex(getGpuPtr(), _iRows, _iCols, (cuDoubleComplex*)ptrCplx);
        if (_cudaStat != cudaSuccess)
        {
            GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
        }

        _ptr->set_ptr(ptrCplx);
        _bComplex = true;

        _cudaStat = cudaZInsert((cuDoubleComplex*)getGpuPtr(), getSize(), (cuDoubleComplex*)gpuData->getGpuPtr(),
                                gpuPos->getGpuPtr(), gpuPos->getSize(), isScalar, &iErr);
    }
    else if (gpuData->isGpuComplex() == false && isGpuComplex())
    {
        _cudaStat = cudaZDInsert((cuDoubleComplex*)getGpuPtr(), getSize(), gpuData->getGpuPtr(),
                                 gpuPos->getGpuPtr(), gpuPos->getSize(), isScalar, &iErr);
    }
    else
    {
        _cudaStat = cudaZInsert((cuDoubleComplex*)getGpuPtr(), getSize(), (cuDoubleComplex*)gpuData->getGpuPtr(),
                                gpuPos->getGpuPtr(), gpuPos->getSize(), isScalar, &iErr);
    }

    if (_cudaStat != cudaSuccess)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)_cudaStat);
    }

    return iErr;
}

PointerCuda::~PointerCuda()
{
    _ptr.reset();
}

