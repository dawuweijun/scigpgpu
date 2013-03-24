extern "C"{

#define IMUL(a, b) __mul24(a, b)

__global__ void scalarProdGPU(
    double *d_C,
	double *d_ACCUM_N,
    double *d_A,
    double *d_B,
    int vectorN,
    int elementN)
{

	for(int vec = blockIdx.x; vec < vectorN; vec += gridDim.x)
	{

		int vectorBase = IMUL(elementN, vec);
		int vectorEnd  = vectorBase + elementN;

		double sum = 0;

		for(int pos = vectorBase + threadIdx.x; pos < vectorEnd; pos += blockDim.x)
			sum += d_A[pos] * d_B[pos];

		d_ACCUM_N[threadIdx.x+blockDim.x*blockIdx.x] = sum;

		if(blockDim.x%2) // if blockDim is odd
			d_ACCUM_N[blockDim.x*blockIdx.x] += d_ACCUM_N[blockDim.x*blockIdx.x+blockDim.x-1];

		for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();
			
			for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x)
				d_ACCUM_N[iAccum+blockDim.x*blockIdx.x] += d_ACCUM_N[iAccum+blockDim.x*blockIdx.x+stride];	

			if((threadIdx.x == 0) && (stride % 2) && (stride != 1))// if stride is odd
					d_ACCUM_N[blockDim.x*blockIdx.x] += d_ACCUM_N[blockDim.x*blockIdx.x+stride-1];
		}
			
		if(threadIdx.x == 0)
				d_C[vec] = d_ACCUM_N[blockDim.x*blockIdx.x];
	}
};
}
