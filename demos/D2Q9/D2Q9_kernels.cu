extern "C"{
  
__global__ void compute_rho(double* rho,
			    double* fIn1,
			    double* fIn2,
			    double* fIn3,
			    double* fIn4,
			    double* fIn5,
			    double* fIn6,
			    double* fIn7,
			    double* fIn8,
			    double* fIn9,
			    int length)
{
  int idx=threadIdx.x+blockIdx.x*blockDim.x;
  if(idx<length)
  {
    rho[idx]=fIn1[idx];
    rho[idx]+=fIn2[idx];
    rho[idx]+=fIn3[idx];
    rho[idx]+=fIn4[idx];
    rho[idx]+=fIn5[idx];
    rho[idx]+=fIn6[idx];
    rho[idx]+=fIn7[idx];
    rho[idx]+=fIn8[idx];
    rho[idx]+=fIn9[idx];
  }
}

__global__ void compute_new_ux_uy(double* ux,double* uy,
			    double* rho,
			    double* fIn1,
			    double* fIn2,
			    double* fIn3,
			    double* fIn4,
			    double* fIn5,
			    double* fIn6,
			    double* fIn7,
			    double* fIn8,
			    double* fIn9,
			    int length)
{
  int idx=threadIdx.x+blockIdx.x*blockDim.x;
 if(idx<length)
 {
   //cx = [  0,   1,  0, -1,  0,    1,  -1,  -1,   1];
   //cy = [  0,   0,  1,  0, -1,    1,   1,  -1,  -1];
   ux[idx]=fIn2[idx];
   ux[idx]-=fIn4[idx];
   uy[idx]=fIn3[idx];
   uy[idx]-=fIn5[idx];
   
   ux[idx]+=fIn6[idx];
   uy[idx]+=fIn6[idx];
   
   ux[idx]-=fIn7[idx];
   uy[idx]+=fIn7[idx];
   
   ux[idx]-=fIn8[idx];
   uy[idx]-=fIn8[idx];
   
   ux[idx]+=fIn9[idx];
   uy[idx]-=fIn9[idx];
   
   ux[idx]/=rho[idx];
   uy[idx]/=rho[idx];
 }
}
  
__global__ void boundary_conditions_inlet(
double* ux,
double* uy,
double* rho,
double* fIn1,
double* fIn2,
double* fIn3,
double* fIn4,
double* fIn5,
double* fIn6,
double* fIn7,
double* fIn8,
double* fIn9,
int start,
int end,
int ld)
{
  int in=0;
  double uMax=0.1;
  double L=98;
  int idx=threadIdx.x+blockIdx.x*blockDim.x+start;
 if(idx<end)
 {
   double y_phys=static_cast<double>(idx)-0.5;
   double tmp2=4*uMax/(L*L) *(y_phys* L- y_phys*y_phys);
   ux[in+idx*ld]=tmp2;
   uy[in+idx*ld]=0;
   double tmp=fIn1[in+idx*ld];
   tmp+=fIn3[in+idx*ld];
   tmp+=fIn5[in+idx*ld];
   tmp+=2*fIn4[in+idx*ld];
   tmp+=2*fIn7[in+idx*ld];
   tmp+=2*fIn8[in+idx*ld];   
   rho[in+idx*ld]=1/(1-tmp2);
   rho[in+idx*ld]*=tmp;
 }
}
  
  
__global__ void boundary_conditions_outlet(
double* ux,
double* uy,
double* rho,
double* fIn1,
double* fIn2,
double* fIn3,
double* fIn4,
double* fIn5,
double* fIn6,
double* fIn7,
double* fIn8,
double* fIn9,
int start,
int end,
int ld)
{
  int out=399;
  int idx=threadIdx.x+blockIdx.x*blockDim.x+start;
 if(idx<end)
 {
   rho[out+idx*ld]=1;
   uy[out+idx*ld]=0;
   
      double tmp=fIn1[out+idx*ld];
   tmp+=fIn3[out+idx*ld];
   tmp+=fIn5[out+idx*ld];
   tmp+=2*fIn2[out+idx*ld];
   tmp+=2*fIn6[out+idx*ld];
   tmp+=2*fIn9[out+idx*ld];   
   
   ux[out+idx*ld]= -1.0+1.0/rho[out+idx*ld]*tmp;
 }
}
  
__global__ void micro_bound_cond_inlet(
double* ux,
double* uy,
double* rho,
double* fIn1,
double* fIn2,
double* fIn3,
double* fIn4,
double* fIn5,
double* fIn6,
double* fIn7,
double* fIn8,
double* fIn9,
int start,
int end,
int ld)
{
    int out=399,in=0;
  int idx=threadIdx.x+blockIdx.x*blockDim.x+start;
 if(idx<end)
 {
  fIn2[in+idx*ld]=fIn4[in+idx*ld] + ux[in+idx*ld] * (2.0/3.0) * rho[in+idx*ld];

  fIn6[in+idx*ld]=fIn8[in+idx*ld] + (1.0/2.0) * ( fIn5[in+idx*ld] - fIn3[in+idx*ld] ) + (1.0/2.0) * rho[in+idx*ld] * uy[in+idx*ld] + (1.0/6.0) * rho[in+idx*ld] * ux[in+idx*ld];

  fIn9[in+idx*ld]=
    fIn7[in+idx*ld]+
    (1.0/2.0) * (fIn3[in+idx*ld] - fIn5[in+idx*ld])
    - (1.0/2.0) * rho[in+idx*ld] * uy[in+idx*ld]
    + (1.0/6.0) * rho[in+idx*ld] * ux[in+idx*ld];
 }
  
}
  
__global__ void micro_bound_cond_outlet(
double* ux,
double* uy,
double* rho,
double* fIn1,
double* fIn2,
double* fIn3,
double* fIn4,
double* fIn5,
double* fIn6,
double* fIn7,
double* fIn8,
double* fIn9,
int start,
int end,
int ld)
{
    int out=399,in=0;
  int idx=threadIdx.x+blockIdx.x*blockDim.x+start;
 if(idx<end)
 {
  fIn4[out+idx*ld]=fIn2[out+idx*ld] -  (2.0/3.0) * rho[out+idx*ld] * ux[out+idx*ld];

  fIn8[out+idx*ld]=
    fIn6[out+idx*ld] 
    + (1.0/2.0) * (fIn3[out+idx*ld] - fIn5[out+idx*ld] )
    - (1.0/2.0) * rho[out+idx*ld] * uy[out+idx*ld]
    - (1.0/6.0) * rho[out+idx*ld] * ux[out+idx*ld];
  fIn7[out+idx*ld]=
    fIn9[out+idx*ld]+
    (1.0/2.0) * (fIn5[out+idx*ld] - fIn3[out+idx*ld])
    + (1.0/2.0) * rho[out+idx*ld] * uy[out+idx*ld]
    - (1.0/6.0) * rho[out+idx*ld] * ux[out+idx*ld];
 }
  
  

}
  
__global__ void streamingstep1
(  double* fIn,
double* fOut,
double* tmp,
int cx,
int cy,
int rows,
int cols)
{
  int ld=rows;
  int idx=threadIdx.x+blockIdx.x*blockDim.x;
  if(idx<cols)
  {
    for(int i=0;i<rows;++i)
    {
    int i2=(i-cx+rows)%rows;
    tmp[i+idx*ld]=fOut[i2+idx*ld];
    }
  }
  
}

__global__ void streamingstep2
(  double* fIn,
double* fOut,
double* tmp,
int cx,
int cy,
int rows,
int cols)
{
    int ld=rows;
  int idx=threadIdx.x+blockIdx.x*blockDim.x;
  if(idx<rows)
  {
    for(int j=0;j<cols;++j)
    {
      int j2=(j-cy+cols)%cols;
      fIn[idx+j*ld]=tmp[idx+j2*ld];
    }
  }
}



__global__ void some_aux_calculations
(double* fIn,
 double* fEq,
 double* fOut,
 double* rho,
 double* ux,
 double* uy,
 double t,
 double omega,
 int cx,
 int cy,
 int length)
 {
     int idx=threadIdx.x+blockIdx.x*blockDim.x;
     if(idx<length)
     {
       double cu=3.0*(cx*ux[idx]+cy*uy[idx]);
       fEq[idx]=rho[idx]*t*(1.0+cu+0.5*cu*cu-1.5*(ux[idx]*ux[idx]+uy[idx]*uy[idx]));
       fOut[idx]=fIn[idx]-omega*(fIn[idx]-fEq[idx]);
       
     }
 }
  
__global__ void bounce(
double* fIn,
double* fOut,
double* obst,
int length	     
){
     int idx=threadIdx.x+blockIdx.x*blockDim.x;
     if(idx<length)
     {
       if(obst[idx]>0.5)
	 fOut[idx]=fIn[idx];
     }
  
}


  
  
__global__ void copykernel(float* dst,double* ux,double* uy, int numberOfElement)
{
  int idx=threadIdx.x+blockIdx.x*blockDim.x;
  if(idx<numberOfElement)
  {
  	float uxf= ux[idx];
  	float uyf= uy[idx];
  	float tmp=uxf * uxf + uyf * uyf;
  	tmp=10*sqrtf(tmp);
  	dst[4*idx]=tmp*tmp;
  	dst[4*idx+1]=0;
  	dst[4*idx+2]=(1.0f-tmp)*(1.0f-tmp);
  	dst[4*idx+3]=1;
  }
}
  
}