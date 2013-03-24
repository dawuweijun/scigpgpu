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

#include <math.h>
#include "idmax.h"

__global__ void idmax_kernel(double* d, int elems, double* result)
{
	int posInGrid  = blockIdx.x  * blockDim.x + threadIdx.x;
   	extern __shared__ double accumResult[];

	if(posInGrid < elems)
	{
		accumResult[threadIdx.x] = d[posInGrid];

		for(int i = posInGrid+blockDim.x*gridDim.x; i < elems; i += blockDim.x*gridDim.x)
		{
				if(d[i] > accumResult[threadIdx.x])
					accumResult[threadIdx.x] = d[i];
		}
		__syncthreads();

		for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
		{
			if(threadIdx.x < stride && posInGrid+stride < elems)
			{
				if(accumResult[stride + threadIdx.x] > accumResult[threadIdx.x])
					accumResult[threadIdx.x] = accumResult[stride + threadIdx.x];
			}
			__syncthreads();
		}

		if(threadIdx.x == 0) result[blockIdx.x] = accumResult[0];
	}
}

cudaError_t cudaIdmax(int elems, double* d, double* res)
{
	double* input  	= NULL;
	double* output 	= NULL;
	cudaError_t cudaStat = cudaGetLastError();

	try
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess) throw cudaStat;

		int threadMax 	= deviceProp.maxThreadsDim[0];
		int blockMax  	= deviceProp.maxGridSize[0];

		int dimgrid  	= (int) ceil((float)elems/threadMax);

		input = d;

		if(blockMax < dimgrid)
			dimgrid = blockMax;

		while(true)
		{
			cudaMalloc((void**)&output,dimgrid*sizeof(double));
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess) throw cudaStat;

			dim3 block(threadMax, 1, 1);
		   	dim3 grid(dimgrid, 1, 1);
		   	idmax_kernel<<<grid, block, threadMax*8>>>(input,elems,output);

            cudaStat = cudaGetLastError();
            if (cudaStat != cudaSuccess) throw cudaStat;

            cudaStat = cudaThreadSynchronize();
			if (cudaStat != cudaSuccess) throw cudaStat;

			if(dimgrid == 1)
				break;

			elems = dimgrid;

			if(input != d)
				cudaFree(input);

			cudaMalloc((void**)&input,elems*sizeof(double));
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess) throw cudaStat;

			cudaMemcpy(input,output,elems*sizeof(double),cudaMemcpyDeviceToDevice);
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess) throw cudaStat;

			cudaFree(output);
			dimgrid = (int) ceil((float)elems/threadMax);
		}

		cudaMemcpy(res,output,sizeof(double),cudaMemcpyDeviceToHost);
		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess) throw cudaStat;

		cudaFree(output);
		if(input != d)
			cudaFree(input);

		return cudaSuccess;
	}
	catch(cudaError_t cudaE)
	{
		if(input != NULL && input != d) cudaFree(input);
		if(output != NULL) cudaFree(output);
		return cudaE;
	}
}
