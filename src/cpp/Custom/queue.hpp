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

#ifndef QUEUE_HPP_
#define QUEUE_HPP_

template<typename ModeDefinition>
class Queue
{
	friend class Context<ModeDefinition>;
	typedef typename ModeDefinition::Context_Handle Context_Handle;
	typedef typename ModeDefinition::Device_Handle Device_Handle;
	typedef typename ModeDefinition::Stream Stream;
	typedef typename ModeDefinition::Status Status;

protected:
	Context_Handle cont;
	Device_Handle dev;
	Queue(Context_Handle,Device_Handle);

public:
	Stream stream;
	Queue();

};

#endif /* QUEUE_HPP_ */
