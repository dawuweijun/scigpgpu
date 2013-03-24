/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2011 - Cédric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/
/* ========================================================================== */
#include "config_gpu.h"
#include "gpu_wrap.h"
#include "useCuda.h"
#include "gw_gpu.h"
#include "checkDevice.h"
#include "gpuContext.hxx"
/* ========================================================================== */
#include "api_scilab.h"
#include "Scierror.h"
#include "sciprint.h"
/* ========================================================================== */
#ifdef WITH_CUDA
    #include <cuda_runtime.h>
    #include <cublas.h>
    #include "makecucomplex.h"
    #include "LU.h"
#endif

#ifdef WITH_OPENCL

#endif
/* ========================================================================== */
int sci_gpuLU(char *fname)
{
    CheckRhs(1,2);
    CheckLhs(2,2);
    #ifdef WITH_CUDA
        cublasStatus status;
    #endif
    SciErr sciErr;
    int*    piAddr_A    = NULL;
    double* h_A         = NULL;
    double* hi_A        = NULL;
    int     rows_A;
    int     cols_A;

    int*    piAddr_Opt  = NULL;
    double* option      = NULL;
    int     rows_Opt;
    int     cols_Opt;

    void*   d_A         = NULL;
    int     na;
    void*   pvPtr       = NULL;

    int     size_A      = sizeof(double);
    bool    bComplex_A  = FALSE;
    int     inputType_A;
    int     inputType_Opt;
    double  res;
    int     posOutput   = 1;

    try
    {
        sciErr = getVarAddressFromPosition(pvApiCtx, 1, &piAddr_A);
        if(sciErr.iErr) throw sciErr;
        if(Rhs == 2)
        {
            sciErr = getVarAddressFromPosition(pvApiCtx, 2, &piAddr_Opt);
            if(sciErr.iErr) throw sciErr;
            sciErr = getVarType(pvApiCtx, piAddr_Opt, &inputType_Opt);
            if(sciErr.iErr) throw sciErr;
            if(inputType_Opt == sci_matrix)
            {
                sciErr = getMatrixOfDouble(pvApiCtx, piAddr_Opt, &rows_Opt, &cols_Opt, &option);
                if(sciErr.iErr) throw sciErr;
            }
            else
                throw "Option syntax is [number,number].";
        }
        else
        {
            rows_Opt=1;
            cols_Opt=2;
            option = (double*)malloc(2*sizeof(double));
            option[0]=0;
            option[1]=0;
        }

        if(rows_Opt != 1 || cols_Opt != 2)
            throw "Option syntax is [number,number].";

        if((int)option[1] == 1 && !isGpuInit())
            throw "gpu is not initialised. Please launch gpuInit() before use this function.";

        sciErr = getVarType(pvApiCtx, piAddr_A, &inputType_A);
        if(sciErr.iErr) throw sciErr;

        #ifdef WITH_CUDA
        if (useCuda())
        {
            if(inputType_A == sci_pointer)
            {
                sciErr = getPointer(pvApiCtx, piAddr_A, (void**)&pvPtr);
                if(sciErr.iErr) throw sciErr;

                gpuMat_CUDA* gmat;
                gmat = static_cast<gpuMat_CUDA*>(pvPtr);
				if(!gmat->useCuda)
					throw "Please switch to OpenCL mode before use this data.";
                rows_A=gmat->rows;
                cols_A=gmat->columns;
                if(gmat->complex)
                {
                    bComplex_A = TRUE;
                    size_A = sizeof(cuDoubleComplex);
                    d_A=(cuDoubleComplex*)gmat->ptr->get_ptr();
                }
                else
                    d_A=(double*)gmat->ptr->get_ptr();

                // Initialize CUBLAS
                status = cublasInit();
                if (status != CUBLAS_STATUS_SUCCESS) throw status;

                na = rows_A * cols_A;
            }
            else if(inputType_A == 1)
            {
                // Get size and data
                if(isVarComplex(pvApiCtx, piAddr_A))
                {
                    sciErr = getComplexMatrixOfDouble(pvApiCtx, piAddr_A, &rows_A, &cols_A, &h_A, &hi_A);
                    if(sciErr.iErr) throw sciErr;
                    size_A = sizeof(cuDoubleComplex);
                    bComplex_A = TRUE;
                }
                else
                {
                    sciErr = getMatrixOfDouble(pvApiCtx, piAddr_A, &rows_A, &cols_A, &h_A);
                    if(sciErr.iErr) throw sciErr;
                }

                na = rows_A * cols_A;

                // Initialize CUBLAS
                status = cublasInit();
                if (status != CUBLAS_STATUS_SUCCESS) throw status;

                // Allocate device memory
                status = cublasAlloc(na, size_A, (void**)&d_A);
                if (status != CUBLAS_STATUS_SUCCESS) throw status;

                // Initialize the device matrices with the host matrices
                if(!bComplex_A)
                {
                    status = cublasSetMatrix(rows_A,cols_A, sizeof(double), h_A, rows_A, (double*)d_A, rows_A);
                    if (status != CUBLAS_STATUS_SUCCESS) throw status;
                }
                else
                    writecucomplex(h_A, hi_A, rows_A, cols_A, (cuDoubleComplex *)d_A);

            }
            else
                throw "Bad argument type.";

            cuDoubleComplex resComplex;
            // Performs operation
            if(!bComplex_A)
                status = decomposeBlockedLU(rows_A, cols_A, rows_A, (double*)d_A, 1);
       //     else
       //         resComplex = cublasZtrsm(na,(cuDoubleComplex*)d_A);

            if (status != CUBLAS_STATUS_SUCCESS) throw status;

            // Put the result in scilab
            switch((int)option[0])
            {
                case 2 :
                case 1 :    sciprint("The first option must be 0 for this function. Considered as 0.\n");

                case 0 :    // Keep the result on the Host.
                {           // Put the result in scilab
                    if(!bComplex_A)
                    {
                        double* h_res = NULL;
                        sciErr=allocMatrixOfDouble(pvApiCtx, Rhs + posOutput, rows_A, cols_A, &h_res);
                        if(sciErr.iErr) throw sciErr;
                        status = cublasGetMatrix(rows_A,cols_A, sizeof(double), (double*)d_A, rows_A, h_res, rows_A);
                        if (status != CUBLAS_STATUS_SUCCESS) throw status;
                    }
                    else
                    {
                        sciErr = createComplexMatrixOfDouble(pvApiCtx, Rhs + posOutput, 1, 1, &resComplex.x,&resComplex.y);
                        if(sciErr.iErr) throw sciErr;
                    }

                    LhsVar(posOutput)=Rhs+posOutput;
                    posOutput++;
                    break;
                }

                default : throw "First option argument must be 0 or 1 or 2.";
            }

            switch((int)option[1])
            {
                case 0 :    // Don't keep the data input on Device.
                {
                    if(inputType_A == sci_matrix)
                    {
                        status = cublasFree(d_A);
                        if (status != CUBLAS_STATUS_SUCCESS) throw status;
                        d_A = NULL;
                    }
                    break;
                }
                case 1 :    // Keep data of the fisrt argument on Device and return the Device pointer.
                {
                    if(inputType_A == sci_matrix)
                    {
                        gpuMat_CUDA* dptr;
                        gpuMat_CUDA tmp={getCudaContext()->genMatrix<double>(getCudaQueue(),rows_A*cols_A),rows_A,cols_A};
                        dptr=new gpuMat_CUDA(tmp);
						dptr->useCuda = true;
                        dptr->ptr->set_ptr((double*)d_A);
                        if(bComplex_A)
                            dptr->complex=TRUE;
                        else
                            dptr->complex=FALSE;

                        sciErr = createPointer(pvApiCtx,Rhs+posOutput, (void*)dptr);
                        if(sciErr.iErr) throw sciErr;
                        LhsVar(posOutput)=Rhs+posOutput;
                    }
                    else
                        throw "The first input argument is already a GPU variable.";

                    posOutput++;
                    break;
                }

                default : throw "Second option argument must be 0 or 1.";
            }
            // Shutdown
            status = cublasShutdown();
            if (status != CUBLAS_STATUS_SUCCESS) throw status;
        }
        #endif

        #ifdef WITH_OPENCL
        if (!useCuda())
        {
            throw "not implemented with OpenCL.";
        }
        #endif
        if(Rhs == 1)
        {
            free(option);
            option = NULL;
        }

        if(posOutput < Lhs+1)
            throw "Too many output arguments.";

        if(posOutput > Lhs+1)
            throw "Too few output arguments.";

        PutLhsVar();
        return 0;
    }
    catch(const char* str)
    {
        Scierror(999,"%s\n",str);
    }
    catch(SciErr E)
    {
        printError(&E, 0);
    }
    #ifdef WITH_CUDA
    catch(cudaError_t cudaE)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)cudaE);
    }
    catch(cublasStatus CublasE)
    {
        GpuError::treat_error<CUDAmode>((CUDAmode::Status)CublasE,1);
    }
    if (useCuda())
    {
        if(inputType_A == 1 && d_A != NULL) cudaFree(d_A);
    }
    #endif
    #ifdef WITH_OPENCL
    if (!useCuda())
    {
        Scierror(999,"not implemented with OpenCL.\n");
    }
    #endif
    if(Rhs == 1 && option != NULL) free(option);
    return EXIT_FAILURE;
}
/* ========================================================================== */

