/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2010 - Vincent LEJEUNE
* 				 
* This file must be used under the terms of the CeCILL.
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at    
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/

#ifndef QUEUE_CUDA_HPP_
#define QUEUE_CUDA_HPP_

template<>
inline Queue<CUDAmode>::Queue()
{

}


template<>
inline Queue<CUDAmode>::Queue(Context_Handle c,Device_Handle d):cont(c),dev(d)
{

}

#endif /* QUEUE_HPP_ */
