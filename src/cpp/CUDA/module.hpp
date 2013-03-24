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

#ifndef MODULE_H_
#define MODULE_H_



template<>
inline Module<CUDAmode>::Module(std::string f,Context_Handle,Device_Handle):isloaded(false),filename(f)
{

}

template<>
inline void Module<CUDAmode>::load()
{
    if(__check_sanity__<CUDAmode>(cuModuleLoad(&mod,filename.c_str())) == -1)
    {
        isloaded = false;
        return;
    }

    isloaded = true;
}

template<>
inline Module<CUDAmode>::Module():isloaded(false),filename("")
{

}

template<>
inline Module<CUDAmode>::Module(const Module& input):isloaded(false),filename(input.filename)
{
    if(input.isloaded)
        load();
}



template<>
inline Kernel<CUDAmode>* Module<CUDAmode>::getFunction(std::string kernelname) const
{
    if (storedfonc.find(kernelname) == storedfonc.end())
    {
        CUfunction tmp;
        if(__check_sanity__<CUDAmode>(cuModuleGetFunction(&tmp,mod,kernelname.c_str())) == -1)
        {
            return NULL;
        }

        Kernel<CUDAmode> p(tmp);
        const_cast<Module&>(*this).storedfonc[kernelname] = p;
    }

    return &const_cast<Module&>(*this).storedfonc[kernelname];
}

template<>
inline Module<CUDAmode>::~Module()
{
    if(isloaded)
        __check_sanity__<CUDAmode>( cuModuleUnload(mod) );
}


#endif /* MODULE_H_ */
