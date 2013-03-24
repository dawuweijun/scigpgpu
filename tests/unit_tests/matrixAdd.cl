#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void matrixAdd(__global double* c, __global const double* a, __global const double* b, int rows, int cols)
{

    int idx = get_global_id(0);

    if (idx < rows*cols)
    {   
    	c[idx] = a[idx] + b[idx];
    }

}
