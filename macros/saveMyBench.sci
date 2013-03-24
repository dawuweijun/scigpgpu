// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

function saveMyBench(fileName, titleGraph, xLabel, yLabel, xData, yData, legendeString)

    export_to_hdf5(fileName,"titleGraph", "xLabel", "yLabel", "xData", "yData", "legendeString");

endfunction

