/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
*           - 2011 - Cedirc Delamarre
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/
#include <string.h>
#include <iostream>
#include <fstream>

#include "builderOpenCL.h"

int Builder::build(char* sourcefile, std::string buildoption)
{
    std::ofstream myfile;
    std::string output=std::string(sourcefile)+std::string(".cl.out");
    myfile.open (output.c_str());

    cl_int ciErrNum = CL_SUCCESS;
    size_t num;
    std::string sourcefilestr = std::string(sourcefile)+std::string(".cl");

    FILE *file = fopen(sourcefilestr.c_str(), "r");
    if(file == NULL)
    	throw GpuError (std::string("Build Failure :\n File not found :\n") +sourcefilestr);
    fclose(file);
   
    unsigned char* code = filetostr(sourcefilestr.c_str(), "", &num);

    cl_program module = clCreateProgramWithSource(Base::cont, 1,(const char**) &code, &num, &ciErrNum);
    __check_sanity__<OPENCLmode>(ciErrNum);

    cl_int status = clBuildProgram(module, 1, &(get_dev()), buildoption.c_str(), 0, 0);
    if (status == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t sz;
        __check_sanity__<OPENCLmode>( clGetProgramBuildInfo(module,get_dev(),CL_PROGRAM_BUILD_LOG,0,NULL,&sz) );
        char* log = new char[sz + 1];
        __check_sanity__<OPENCLmode>( clGetProgramBuildInfo(module,get_dev(),CL_PROGRAM_BUILD_LOG,sz,reinterpret_cast<void*>(log),NULL) );
        std::string slog=std::string(log);
        delete [] log;
        throw GpuError (std::string("Build Failure :\n") + slog);
    }

    size_t binlength;
    __check_sanity__<OPENCLmode>(clGetProgramInfo(module,CL_PROGRAM_BINARY_SIZES,sizeof(size_t),&binlength,NULL));
    unsigned char* bin=new unsigned char[binlength+1];
    __check_sanity__<OPENCLmode>(clGetProgramInfo(module,CL_PROGRAM_BINARIES,binlength*sizeof(unsigned char),&bin,NULL));
    bin[binlength]='\0';
    std::string retvalue=std::string(reinterpret_cast<char*>(bin));
    delete [] code;

    myfile << retvalue;
    myfile.close();

    return 0;
}
