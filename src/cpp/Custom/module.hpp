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

#ifndef MODULE_HPP_
#define MODULE_HPP_

#include <map>

template<typename ModeDefinition>
class Module
{
	typedef typename ModeDefinition::Status Status;
	typedef typename ModeDefinition::Context_Handle Context_Handle;
	typedef typename ModeDefinition::Module_Handle Module_Handle;
	typedef typename ModeDefinition::Device_Handle Device_Handle;
	friend class Context<ModeDefinition>;
protected:
	Context_Handle cont;
	Device_Handle dev;
	Module_Handle mod;
	bool isloaded;
	std::string filename;
	std::map<std::string, Kernel<ModeDefinition> > storedfonc;

	Module(std::string f,Context_Handle c=NULL,Device_Handle d=NULL);
	void load();
public:
	Module();
	Module(const Module& input);
	Kernel<ModeDefinition>* getFunction(std::string kernelname) const;
	~Module();
};


#endif /* MODULE_HPP_ */
