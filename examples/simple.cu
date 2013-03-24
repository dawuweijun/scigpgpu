extern "C"
{

__global__ void someSimpleKernel(double* src,double* dst, int numberOfElement)
{
  int idx=threadIdx.x+blockIdx.x*blockDim.x;
  if(idx<numberOfElement)
  {
  	dst[idx]=src[idx];
  }
}

__global__ void copykernel(float* dst,double* ux,double* uy, int numberOfElement)
{
  int idx=threadIdx.x+blockIdx.x*blockDim.x;
  if(idx<numberOfElement)
  {
  	dst[4*idx]=(float)ux[idx]/256.0f;
  	dst[4*idx+1]=(float)uy[idx]/256.0f;
  	dst[4*idx+2]=1;
  	dst[4*idx+3]=1;
  }
}

__global__ void add(double* dst,double* src, int numberOfElement)
{
  int idx=threadIdx.x+blockIdx.x*blockDim.x;
  if(idx<numberOfElement)
  {
  	dst[idx]=2 * src[idx];
  }
}

}