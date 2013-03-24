// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

//==============================================================================
// Benchmark for multiple product on CPU
//==============================================================================

// <-- BENCH NB RUN : 10 -->

stacksize('max');

A=rand(1000,1000);
B=rand(1000,1000);
n = 100

// <-- BENCH START -->

cpu=A*B;

for i = 0:n,
    cpu=cpu*B;
    cpu=A*cpu;
end;

cpu=cpu*B;
cpu=A*cpu;

// <-- BENCH END -->

