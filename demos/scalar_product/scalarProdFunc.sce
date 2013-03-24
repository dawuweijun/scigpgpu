function ret=scalarProduct(a,b)

	if size(a,"r") == size(b,"r")
	if size(a,"c") == size(b,"c")

//		if size(a,"c") < size(a,"r")
//			a=a';
//			b=b';
//		end

		if size(a,"c") >= size(a,"r")

			if modulo(size(a,"c"),size(a,"r")) == 0

				vector_N=size(a,"c")/size(a,"r")
				block_N=vector_N;
				if block_N> 65535
					block_N = 65535;
				end

				element_N=size(a,"*")/vector_N
				thread_N = element_N
				if thread_N > 512
					thread_N = 512;
				end

				d_a=gpuSetData(a);
				d_b=gpuSetData(b);
		
				d_c=gpuAlloc(1,vector_N); // cudaAlloc(1,number of vectors)

				d_d=gpuAlloc(1,element_N*vector_N);

				bin=gpuBuild("scalarProd_kernel");
				fonc=gpuLoadFunction(bin,"scalarProdGPU");

				lst=list(d_c,d_d,d_a,d_b,int32(vector_N),int32(element_N)); 
								//list(d_c,d_d,d_a,d_b,
								//number of vectors,
								//number of elements of each vectors 


				gpuApplyFunction(fonc,lst,1,thread_N,1,block_N);
				//fonc,args,block_height,block_width,grid_height,grid_width
								// block_width = number of elements
								// block_height = 1
								// grid_width = number of vectors
								// grid_height = 1

				ret=gpuGetData(d_c)

				gpuFree(d_c);
				gpuFree(d_b);
				gpuFree(d_a);
				gpuFree(d_d);

			else
				ret = "Matrix must be square"
			end
			
		else
			ret="Tranpose the two matrix and relaunch the function."
		end
	end
	else
		ret="Matrix size error"
	end
endfunction

