h_a=[1,2,3;4,5,6]

d_a=gpuSetData(h_a);
d_c=gpuAlloc(3,2);

abs_path=get_absolute_file_path("transpose.sce");

bin=gpuBuild(abs_path+"transpose_kernel");
fonc=gpuLoadFunction(bin,"transpose_naive");

lst=list(d_c,d_a,int32(3),int32(2));

gpuApplyFunction(fonc,lst,2,3,1,1); // fonc,lst,block_h,block_w,grid_h,grid_w

h_c=gpuGetData(d_c)

gpuFree(d_c);
gpuFree(d_a);
clear h_c;
clear h_a;

