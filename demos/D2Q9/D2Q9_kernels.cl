#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void compute_rho(__global double* rho,
			    __global double* fIn1,
			    __global double* fIn2,
			    __global double* fIn3,
			    __global double* fIn4,
			    __global double* fIn5,
			    __global double* fIn6,
			    __global double* fIn7,
			    __global double* fIn8,
			    __global double* fIn9,
			    int length)
{
  int idx=get_global_id(0);
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

__kernel void compute_new_ux_uy(__global double* ux,__global double* uy,
			    __global double* rho,
			    __global double* fIn1,
			    __global double* fIn2,
			    __global double* fIn3,
			    __global double* fIn4,
			    __global double* fIn5,
			    __global double* fIn6,
			    __global double* fIn7,
			    __global double* fIn8,
			    __global double* fIn9,
			    int length)
{
  int idx=get_global_id(0);
 if(idx<length)
 {
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
  
__kernel void boundary_conditions_inlet(
__global double* ux,
__global double* uy,
__global double* rho,
__global double* fIn1,
__global double* fIn2,
__global double* fIn3,
__global double* fIn4,
__global double* fIn5,
__global double* fIn6,
__global double* fIn7,
__global double* fIn8,
__global double* fIn9,
int start,
int end,
int ld)
{
  int in=0;
  double uMax=0.1;
  double L=98;
    int idx=get_global_id(0)+start;
 if(idx<end)
 {
   double y_phys=(double)(idx)-0.5;
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
  
  
__kernel void boundary_conditions_outlet(
__global double* ux,
__global double* uy,
__global double* rho,
__global double* fIn1,
__global double* fIn2,
__global double* fIn3,
__global double* fIn4,
__global double* fIn5,
__global double* fIn6,
__global double* fIn7,
__global double* fIn8,
__global double* fIn9,
int start,
int end,
int ld)
{
  int out=399;
    int idx=get_global_id(0)+start;
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
  
__kernel void micro_bound_cond_inlet(
__global double* ux,
__global double* uy,
__global double* rho,
__global double* fIn1,
__global double* fIn2,
__global double* fIn3,
__global double* fIn4,
__global double* fIn5,
__global double* fIn6,
__global double* fIn7,
__global double* fIn8,
__global double* fIn9,
int start,
int end,
int ld)
{
    int out=399,in=0;
    int idx=get_global_id(0)+start;
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
  
__kernel void micro_bound_cond_outlet(
__global double* ux,
__global double* uy,
__global double* rho,
__global double* fIn1,
__global double* fIn2,
__global double* fIn3,
__global double* fIn4,
__global double* fIn5,
__global double* fIn6,
__global double* fIn7,
__global double* fIn8,
__global double* fIn9,
int start,
int end,
int ld)
{
    int out=399,in=0;
    int idx=get_global_id(0)+start;
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
  
__kernel void streamingstep1
(  __global double* fIn,
__global double* fOut,
__global double* tmp,
int cx,
int cy,
int rows,
int cols)
{
  int ld=rows;
    int idx=get_global_id(0);
  if(idx<cols)
  {
    for(int i=0;i<rows;++i)
    {
    int i2=(i-cx+rows)%rows;
    tmp[i+idx*ld]=fOut[i2+idx*ld];
    }
  }
  
}

__kernel void streamingstep2
(  __global double* fIn,
__global double* fOut,
__global double* tmp,
int cx,
int cy,
int rows,
int cols)
{
    int ld=rows;
   int idx=get_global_id(0);
  if(idx<rows)
  {
    for(int j=0;j<cols;++j)
    {
      int j2=(j-cy+cols)%cols;
      fIn[idx+j*ld]=tmp[idx+j2*ld];
    }
  }
}



__kernel void some_aux_calculations
(__global double* fIn,
 __global double* fEq,
 __global double* fOut,
 __global double* rho,
 __global double* ux,
 __global double* uy,
 double t,
 double omega,
 int cx,
 int cy,
 int length)
 {
       int idx=get_global_id(0);
     if(idx<length)
     {
       double cu=3.0*(cx*ux[idx]+cy*uy[idx]);
       fEq[idx]=rho[idx]*t*(1.0+cu+0.5*cu*cu-1.5*(ux[idx]*ux[idx]+uy[idx]*uy[idx]));
       fOut[idx]=fIn[idx]-omega*(fIn[idx]-fEq[idx]);
       
     }
 }
  
__kernel void bounce(
__global double* fIn,
__global double* fOut,
__global double* obst,
int length	     
){
       int idx=get_global_id(0);
     if(idx<length)
     {
       if(obst[idx]>0.5)
	 fOut[idx]=fIn[idx];
     }
  
}


__kernel void cast_tst(__global double* sample, int length)
{
         int idx=get_global_id(0);
     if(idx<length)
     {
       float tmp=sample[idx];
       sample[idx]=2*tmp+1.0;
     }
}
