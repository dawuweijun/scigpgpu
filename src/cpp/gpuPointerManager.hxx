/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) DIGITEO - 2011 - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/

#ifndef __GPUPOINTERMANAGER_HXX__
#define __GPUPOINTERMANAGER_HXX__

#include <vector>
#include "gpuPointer.hxx"
#include "dynlib_gpu.h"

class GPU_IMPEXP PointerManager
{
    public :
                static PointerManager* getInstance(void);
                static void killInstance(void);

                bool addGpuPointerInManager(GpuPointer* pGpuPointer);
                bool removeGpuPointerInManager(GpuPointer* pGpuPointer);
                bool findGpuPointerInManager(GpuPointer* pGpuPointer);
                int getPositionGpuPointerInManager(GpuPointer* pGpuPointer);
                GpuPointer* getLastGpuPointerInManager(void);

    private :
                PointerManager();
                ~PointerManager();

                static PointerManager* _instance;
                std::vector<GpuPointer*> _gpuPointerVector;

};

#endif //__GPUPOINTERMANAGER_HXX__
