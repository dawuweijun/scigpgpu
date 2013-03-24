// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

//==============================================================================
// Benchmark for complex product on GPU and CPU
//==============================================================================


stacksize('max');
tmp=ones(6,2);

k=1000;
m=10;
n=10;

a=rand(m,k)+rand(m,k)*%i;
b=rand(k,n)+rand(k,n)*%i;
tic();gpuRes=gpuMult(a,b); res=gpuGetData(gpuRes); gpuFree(gpuRes); tmp(1,1)=toc();
tic();cpuRes=a*b;           tmp(1,2)=toc();

k=1000;
m=5000;
n=10;
a=rand(m,k)+rand(m,k)*%i;
b=rand(k,n)+rand(k,n)*%i;
tic();gpuRes=gpuMult(a,b); res=gpuGetData(gpuRes); gpuFree(gpuRes);  tmp(2,1)=toc();
tic();cpuRes=a*b;           tmp(2,2)=toc();

k=10;
m=1000;
n=2000;
a=rand(m,k)+rand(m,k)*%i;
b=rand(k,n)+rand(k,n)*%i;
tic();gpuRes=gpuMult(a,b); res=gpuGetData(gpuRes); gpuFree(gpuRes);  tmp(3,1)=toc();
tic();cpuRes=a*b;           tmp(3,2)=toc();

k=10000;
m=100;
n=100;
a=rand(m,k)+rand(m,k)*%i;
b=rand(k,n)+rand(k,n)*%i;
tic();gpuRes=gpuMult(a,b); res=gpuGetData(gpuRes); gpuFree(gpuRes);  tmp(4,1)=toc();
tic();cpuRes=a*b;           tmp(4,2)=toc();

k=20;
m=2000;
n=2000;
a=rand(m,k)+rand(m,k)*%i;
b=rand(k,n)+rand(k,n)*%i;
tic();gpuRes=gpuMult(a,b); res=gpuGetData(gpuRes); gpuFree(gpuRes);  tmp(5,1)=toc();
tic();cpuRes=a*b;           tmp(5,2)=toc();;

k=10000;
m=1000;
n=1000;
a=rand(m,k)+rand(m,k)*%i;
b=rand(k,n)+rand(k,n)*%i;
tic();gpuRes=gpuMult(a,b); res=gpuGetData(gpuRes); gpuFree(gpuRes);  tmp(6,1)=toc();
tic();cpuRes=a*b;           tmp(6,2)=toc()

clear res;
clear a;
clear b;
