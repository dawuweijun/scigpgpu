function getNVCC_PATH()
	if ~isdef("NVCC_PATH") then
		if getos() == "Windows"
			pathCudaBin = getenv("CUDA_BIN_PATH","");
		    	install = "";
			if pathCudaBin == "" then
				install = "Please install CUDA Toolkit 3.2 or more."+ascii(10);
		    	end
		    	if ~haveacompiler() then
		        	install = install + "Please install Visual Studio compiler."+ascii(10);
		    	end
		    	if install == "" then
				NVCC_PATH = pathCudaBin;
		    	else
		        	error(install);
		    	end
		end

		if getos() == "Linux"
			path = mopen(gpgpu_getToolboxPath()+'/nvccdir.txt','rt');
			NVCC_PATH = mfscanf(path,'%s');
			mclose(path);
			if ~isfile(NVCC_PATH+"/nvcc") then
				error("Please, check that you have correctly:"+ascii(10)+ascii(10)+"    1) Install the CUDA Toolkit 3.2 or more."+ascii(10)+"    2) Install the g++ compiler."+ascii(10)+"    3) Add the nvcc path in nvccdir.txt"+ascii(10));
			end
		end
	else
		error("NVCC_PATH is already define : " + NVCC_PATH);
	end

	NVCC_PATH = resume(NVCC_PATH);
endfunction
