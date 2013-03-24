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
#ifndef DEVICE_HPP_
#define DEVICE_HPP_


/*!
 * Device represents a GPGPU capable device present in the system.
 */
template<class ModeDefinition>
class Device
{
	enum
	{
		mode=ModeDefinition::mode
	};
	typedef typename ModeDefinition::Device_Handle Device_Handle;
	typedef typename ModeDefinition::Status Status;
	typedef typename ModeDefinition::Device_identifier Device_identifier;
	friend class Context<ModeDefinition> ;
protected:
	Device_Handle dev;
	std::pair<int, int> dev_cap;
	size_t mem;
	bool support_plm;
	bool support_cce;
	std::string name;

	Device();
	int initDevice(Device_identifier ordinal);

public:
	/*!
	 * Return the CUDA device_capability of the device (for instance to know if it can handle double).
	 * Does not currently work in OpenCL mode.
	 */
	std::pair<int, int>	device_capability() const;
	/*!
	 * Return total amount of dedicated memory on the device
	 */
	int	memory_amount() const;

	/*!
	 * Return the name of the device
	 */
	std::string	get_name() const;

	/*!
	 * True if the device does support page_locked_memory feature, also know as "pinned memory".
	 * Does not currently work in OpenCL mode.
	 */
	bool support_page_locked_memory() const;

	/*!
	 * True if the device does support concurrent copy and execution feature, also know as "overlapping".
	 * Does not currently work in OpenCL mode.
	 */
	bool support_concurrent_copy_execution() const;
};


/*!
 * \cond
 */
template<typename ModeDefinition>
inline
std::pair<int, int>	Device<ModeDefinition>::device_capability() const
{
	return dev_cap;
}

template<typename ModeDefinition>
inline
int	Device<ModeDefinition>::memory_amount() const
{
	return mem;
}

template<typename ModeDefinition>
inline
std::string	Device<ModeDefinition>::get_name() const
{
	return name;
}

template<typename ModeDefinition>
inline
bool Device<ModeDefinition>::support_page_locked_memory() const
{
	return support_plm;
}

template<typename ModeDefinition>
inline
bool Device<ModeDefinition>::support_concurrent_copy_execution() const
{
	return support_cce;
}

/*!
 * \endcond
 */


#endif /* DEVICE_HPP_ */
