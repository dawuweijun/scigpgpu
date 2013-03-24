stacksize('max');
tmp=ones(4,5);

a=rand(1000,1000);

tic();gpuRes=gpuMax(a); tmp(1,1)=toc();
tic();cpuRes=max(a);    tmp(1,2)=toc();

tmp(1,3) = gpuRes;
tmp(1,4) = cpuRes;
tmp(1,5) = gpuRes - cpuRes;

a=rand(10000,1000);

tic();gpuRes=gpuMax(a); tmp(2,1)=toc();
tic();cpuRes=max(a);    tmp(2,2)=toc();

tmp(2,3) = gpuRes;
tmp(2,4) = cpuRes;
tmp(2,5) = gpuRes - cpuRes;

a=rand(5000,5000);

tic();gpuRes=gpuMax(a); tmp(3,1)=toc();
tic();cpuRes=max(a);    tmp(3,2)=toc();

tmp(3,3) = gpuRes;
tmp(3,4) = cpuRes;
tmp(3,5) = gpuRes - cpuRes;

a=rand(10000,5000);

tic();gpuRes=gpuMax(a);	tmp(4,1)=toc();
tic();cpuRes=max(a);    tmp(4,2)=toc();

tmp(4,3) = gpuRes;
tmp(4,4) = cpuRes;
tmp(4,5) = gpuRes - cpuRes

clear a;
