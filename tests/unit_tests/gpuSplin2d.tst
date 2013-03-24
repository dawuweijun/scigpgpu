// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) Scilab-Enterprises - 2013 - Cedric Delamarre
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

// <-- ENGLISH IMPOSED -->

for n = [2 3 7]// a n x n interpolation grid
    x = linspace(0,2*%pi,n); y = x;
    z = cos(x')*cos(y);
    for splinType = ["monotone", "fast", "fast_periodic", ..
                     "not_a_knot", "natural", "periodic"]
        C = splin2d(x, y, z, splinType);
        dC = gpuSplin2d(x, y, z, splinType);
        assert_checkalmostequal(C, gpuGetData(dC), [], 10*%eps);
        gpuFree(dC);
    end
end
