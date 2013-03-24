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

#ifndef KERNEL_HPP_
#define KERNEL_HPP_

#include <memory> // std::shared_ptr

#define ALIGN_UP(off,alignment) \
  (off) = ( (off) +  ( alignment ) - 1 ) & ~ ( ( alignment ) - 1 );

/*!
 * This classe represents a kernel, that is a on GPU executed function.
 * It should not be created directly, but obtained by the getFunction Module's member.
 */
template<typename ModeDefinition>
class Kernel
{
	typedef typename ModeDefinition::Status Status;
	typedef typename ModeDefinition::Function_Handle Function_Handle;
	typedef typename ModeDefinition::Stream Stream;
protected:

	int offset;
	Function_Handle fonc;

public:
	Kernel();
	Kernel(Function_Handle fptr);

	/*!
	 * Pass a Matrix<T> as an argument.
	 */
	template<typename T>
	void pass_argument(std::shared_ptr<Matrix<ModeDefinition,T> > input);

	/*!
	 * Pass a GLMatrix as an argument.
	 * An OpenGL context must be set and active.
	 */
	template<typename T>
	void pass_argument(std::shared_ptr<GLMatrix<ModeDefinition, T> > input);

	/*!
	 * Pass a scalar T (int, float, double) as an input
	 */
	template<typename T>
	void pass_argument(T f);

	/*!
	 * Plan an execution of the kernel in the queue.
	 */
	void launch(Queue<ModeDefinition> queue,int xdim, int ydim, int grid_w, int grid_h);
};



#endif /* KERNEL_HPP_ */
