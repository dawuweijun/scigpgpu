abs_path=get_absolute_file_path("simple.sce");

bin = gpuBuild(abs_path+"simple");

A_host=rand(256,256);
A=gpuSetData(A_host);
B=gpuAlloc(256,256);

Kernel=gpuLoadFunction(bin,"someSimpleKernel");
lst=list(A,B,int32(256*256));
gpuApplyFunction(Kernel,lst,128,1,256*256/128,1);
B_host=gpuGetData(B);

clear Kernel;
clear lst;

