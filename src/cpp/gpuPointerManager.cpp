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

#include "gpuPointerManager.hxx"

PointerManager* PointerManager::_instance = NULL;

PointerManager* PointerManager::getInstance()
{
    if (!_instance)
    {
        _instance = new PointerManager();
    }
    return _instance;
}
void PointerManager::killInstance(void)
{
    if (_instance)
    {
        delete _instance;
        _instance = NULL;
    }
}

PointerManager::PointerManager()
{

}

bool PointerManager::addGpuPointerInManager(GpuPointer* pGpuPointer)
{
    if (pGpuPointer)
    {
        if (!findGpuPointerInManager(pGpuPointer))
        {
            _gpuPointerVector.push_back(pGpuPointer);
        }
        return true;
    }
    return false;
}

bool PointerManager::removeGpuPointerInManager(GpuPointer* pGpuPointer)
{
    if (pGpuPointer)
    {
        int pos = getPositionGpuPointerInManager(pGpuPointer);
        if (pos > -1)
        {
            _gpuPointerVector.erase(_gpuPointerVector.begin() + pos);
            return true;
        }
    }
    return false;
}
bool PointerManager::findGpuPointerInManager(GpuPointer* pGpuPointer)
{
    if (!_gpuPointerVector.empty())
    {
        int pos = getPositionGpuPointerInManager(pGpuPointer);
        if (pos > -1)
        {
            return true;
        }
    }
    return false;
}
int PointerManager::getPositionGpuPointerInManager(GpuPointer* pGpuPointer)
{
    int pos = -1;
    if (pGpuPointer)
    {
        for (size_t i = 0; i < _gpuPointerVector.size(); i++)
        {
            if (_gpuPointerVector[i] == pGpuPointer)
            {
                return i;
            }
        }
    }
    return pos;
}
GpuPointer* PointerManager::getLastGpuPointerInManager(void)
{
    GpuPointer* pGpuPointer = NULL;
    if (_gpuPointerVector.size() > 0)
    {
        pGpuPointer = _gpuPointerVector[_gpuPointerVector.size() - 1];
    }
    return pGpuPointer;
}

PointerManager::~PointerManager()
{
    _gpuPointerVector.clear();
}

