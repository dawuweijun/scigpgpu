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

#ifndef ERROR_HPP_
#define ERROR_HPP_

#include <exception>
#include <iostream>

class GpuError: public std::exception
{
protected:
	int err_num;
	std::string err_msg;
public:
	GpuError(std::string msg, int id = 0);
	virtual const char* what() const throw ();
	const int getId();
	~GpuError() throw ();
	template<typename ModeDefinition>
	static int treat_error(typename ModeDefinition::Status id, int who=0);
};

inline GpuError::GpuError(std::string msg, int id) :
	err_num(id), err_msg(msg)
{

}

inline const char* GpuError::what() const throw ()
{
	return err_msg.c_str();
}
inline const int GpuError::getId()
{
	return err_num;
}
inline GpuError::~GpuError() throw ()
{

}

#endif /* ERROR_HPP_ */
