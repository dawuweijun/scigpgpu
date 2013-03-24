h_a=[1 2 3 3 4 5 6 7 4 5 6 5 4 9 8 3;1 5 9 1 7 2 3 6 3 5 1 4 9 5 2 3]
h_b=[1 5 9 1 7 2 3 6 3 5 1 4 9 5 2 3;1 2 3 3 4 5 6 7 4 5 6 5 4 9 8 3]

d_a=gpuSetData(h_a);
d_b=gpuSetData(h_b);
d_c=gpuAlloc(1,2); // gpuAlloc(1,number of vectors)
d_d=gpuAlloc(1,16*2); // (1,element_N*vector_N)

abs_path=get_absolute_file_path("scalarProd.sce");

bin=gpuBuild(abs_path + "scalarProd_kernel");
fonc=gpuLoadFunction(bin,"scalarProdGPU");

lst=list(d_c,d_d,d_a,d_b,int32(2),int32(16)); 		// list(d_c,d_a,d_b,number of vectors,number of elements of each vectors);
gpuApplyFunction(fonc,lst,1,16,1,2);	//fonc,args,block_height,block_width,grid_height,grid_width
								// block_width = number of elements
								// block_height = 1
								// grid_width = number of vectors
								// grid_height = 1
h_c=gpuGetData(d_c)

gpuFree(d_c);
gpuFree(d_b);
gpuFree(d_a);

