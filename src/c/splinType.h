/*
* Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
* Copyright (C) Scilab Enterprises - 2013 - Cedric DELAMARRE
*
* This file must be used under the terms of the CeCILL.
* This source file is licensed as described in the file COPYING, which
* you should have received as part of this distribution.  The terms
* are also available at
* http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
*
*/
/* ==================================================================== */
#ifndef __SPLINTYPE_H__
#define __SPLINTYPE_H__

#ifdef __cplusplus
extern "C" {
#endif
    enum SplineType {
        NOT_A_KNOT,
        NATURAL,
        CLAMPED,
        PERIODIC,
        MONOTONE,
        FAST,
        FAST_PERIODIC
    };
#ifdef __cplusplus
}
#endif /* extern "C" */


#endif /* __SPLINTYPE_H__ */
/* ==================================================================== */
