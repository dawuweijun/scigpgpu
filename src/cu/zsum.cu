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
#include <stdio.h>
#include <cuComplex.h>
#include "zsum.h"

__global__ void zsum_kernel(cuDoubleComplex* d, int elems, cuDoubleComplex* result)
{
	int posInGrid  = blockIdx.x  * blockDim.x + threadIdx.x;
   	extern __shared__ cuDoubleComplex accumResult[];

	accumResult[threadIdx.x].x = 0.0;
	accumResult[threadIdx.x].y = 0.0;

	for(int i = posInGrid; i < elems; i += blockDim.x*gridDim.x)
	{
		accumResult[threadIdx.x].x += d[i].x;
		accumResult[threadIdx.x].y += d[i].y;
	}
	__syncthreads();

	for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
	{
		if(threadIdx.x < stride)
		{
			accumResult[threadIdx.x].x += accumResult[stride + threadIdx.x].x;
			accumResult[threadIdx.x].y += accumResult[stride + threadIdx.x].y;
		}
		__syncthreads();
	}

	if(threadIdx.x == 0)
	{
		result[blockIdx.x].x = accumResult[0].x;
		result[blockIdx.x].y = accumResult[0].y;
	}
}

cudaError_t cudaZsum(int elems, cuDoubleComplex* d, cuDoubleComplex* res)
{
	cuDoubleComplex* input	= NULL;
	cuDoubleComplex* output	= NULL;
	cudaError_t cudaStat = cudaGetLastError();

	try
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess) throw cudaStat;

		int threadMax = deviceProp.maxThreadsDim[0];
		int blockMax  = deviceProp.maxGridSize[0];

		int dimgrid  	= (int) ceil((float)elems/threadMax);

		input = d;

		if(blockMax < dimgrid)
			dimgrid = blockMax;

		while(true)
		{
			cudaMalloc((void**)&output,dimgrid*sizeof(cuDoubleComplex));
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess) throw cudaStat;

			dim3 block(threadMax, 1, 1);
		   	dim3 grid(dimgrid, 1, 1);
		   	zsum_kernel<<<grid, block, threadMax*16>>>(input,elems,output);

            cudaStat = cudaGetLastError();
            if (cudaStat != cudaSuccess) throw cudaStat;

            cudaStat = cudaThreadSynchronize();
			if (cudaStat != cudaSuccess) throw cudaStat;

			if(dimgrid == 1)
				break;

			elems = dimgrid;

			if(input != d)
				cudaFree(input);

			cudaMalloc((void**)&input,elems*sizeof(cuDoubleComplex));
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess) throw cudaStat;

			cudaMemcpy(input,output,elems*sizeof(cuDoubleComplex),cudaMemcpyDeviceToDevice);
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess) throw cudaStat;

			cudaFree(output);
			dimgrid = (int) ceil((float)elems/threadMax);
		}

		cudaMemcpy(res,output,sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
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
