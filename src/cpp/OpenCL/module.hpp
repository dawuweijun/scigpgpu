/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/

#ifndef MODULE_OPENCL_H_
#define MODULE_OPENCL_H_

#include <cstdio>
#include <cstring>

inline unsigned char*
filetostr(const char* cFilename, const char* cPreamble,
    size_t* szFinalLength)
{

  FILE* file = NULL;
  size_t srclength;

  file = fopen(cFilename, "rb");
  if (file == 0)
    {
      return NULL;
    }

  size_t preamblelength = strlen(cPreamble);

  fseek(file, 0, SEEK_END);
  srclength = ftell(file);
  fseek(file, 0, SEEK_SET);
  unsigned char* cSourceString = new unsigned char[srclength + preamblelength + 1];
  memcpy(cSourceString, cPreamble, preamblelength);
  if (fread((cSourceString) + preamblelength, srclength, 1, file) != 1)
    {
      fclose(file);
      delete [] cSourceString;
      return 0;
    }

  fclose(file);
  if (szFinalLength != 0)
    {
      *szFinalLength = srclength + preamblelength;
    }
  cSourceString[srclength + preamblelength] = '\0';

  return cSourceString;
}


template<>
inline Module<OPENCLmode>::Module(std::string f,Context_Handle c,Device_Handle d):isloaded(false),filename(f),cont(c),dev(d)
{
	  cl_int ciErrNum = CL_SUCCESS,binStatus = CL_SUCCESS;
	  size_t num;
	  unsigned char* code = filetostr(filename.c_str(), "", &num);

	  mod = clCreateProgramWithBinary(c, 1,&dev,&num,(const unsigned char**)&code,&binStatus,&ciErrNum);
	  __check_sanity__<OPENCLmode>(binStatus);
	  __check_sanity__<OPENCLmode>(ciErrNum);
	  delete [] code;

}

template<>
inline void Module<OPENCLmode>::load()
{
  cl_int status = clBuildProgram(mod, 1, &dev, "-Werror", 0, 0);
  if (status == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t sz;
    __check_sanity__<OPENCLmode> ( clGetProgramBuildInfo(mod,dev,CL_PROGRAM_BUILD_LOG,0,NULL,&sz) );
    char* log = new char[sz + 1];
    __check_sanity__<OPENCLmode> ( clGetProgramBuildInfo(mod,dev,CL_PROGRAM_BUILD_LOG,sz,reinterpret_cast<void*>(log),NULL) );
    std::string slog=std::string(log);
    delete [] log;
    throw GpuError(std::string("Build Failure :\n") + slog);
  }

	isloaded=true;
}

template<>
inline Module<OPENCLmode>::Module():isloaded(false),filename("")
{

}

template<>
inline Module<OPENCLmode>::Module(const Module& input):isloaded(false),filename(input.filename)
{
	if(input.isloaded)
		load();
}

template<>
inline Kernel<OPENCLmode>* Module<OPENCLmode>::getFunction(std::string kernelname) const
{
	  if (storedfonc.find(kernelname) == storedfonc.end())
	    {
		  cl_int ciErrNum = CL_SUCCESS;
	      Kernel<OPENCLmode> ker(clCreateKernel(mod, kernelname.c_str(), &ciErrNum));
	      __check_sanity__<OPENCLmode>(ciErrNum);
	      const_cast<Module&>(*this).storedfonc.insert(std::pair<std::string, Kernel<OPENCLmode> > (kernelname, ker));
	    }
	  return &const_cast<Module&>(*this).storedfonc[kernelname];
}

template<>
inline Module<OPENCLmode>::~Module()
{

}

#endif /* MODULE_H_ */
