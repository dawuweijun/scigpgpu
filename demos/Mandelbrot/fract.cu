extern "C"{
__global__ void computeFract(double* xvect, double* yvect, int xsize, int ysize, int nmax, double* output)
{

    int i  = blockIdx.x  * blockDim.x + threadIdx.x;
    int j  = blockIdx.y  * blockDim.y + threadIdx.y;
    double x, xtemp, y, x0, y0, xx, yy;
    int k = 0;

    x  = xvect[i];
    y  = yvect[j];
    x0 = x;
    y0 = y;
    xx = x*x;
    yy = y*y;

    while((xx + yy) < 4 && k < nmax )
    {
        y   = 2 * x * y + y0;
        x   = xx - yy + x0;
        xx  = x * x;
        yy  = y * y;
        k++;
    }

    output[ j + i * xsize ] = (double)k;
}
}
