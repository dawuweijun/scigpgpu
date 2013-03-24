extern "C"
__global__ void
matrixAdd( double* C, double* A, double* B, int M, int N)
{
    int idx = blockIdx.x;
    int idy = blockIdx.y;
    
    int dx = blockDim.x;
    int dy = blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int x=tx+dx*idx;
    int y=ty+dy*idy;

    if(x<M && y<N)
      C[ x + y*M ]= A[ x + y*M ] + B[ x+ y*M ];
}
