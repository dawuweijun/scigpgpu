//
//  Copyright (C) 2010 - 2011 - DIGITEO - Allan CORNET
//					  Cedric Delamarre
//  This file must be used under the terms of the CeCILL.
//  This source file is licensed as described in the file COPYING, which
//  you should have received as part of this distribution.  The terms
//  are also available at
//  http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
//

// Returns the path to the current module.
function path = gpgpu_getToolboxPath()
  [fs, path] = libraryinfo("sciGPGPUlib");
  path = pathconvert(fullpath(path + "../"), %t, %t);
endfunction
