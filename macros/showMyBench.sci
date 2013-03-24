// =============================================================================
// Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
// Copyright (C) 2011 - DIGITEO - Cedric DELAMARRE
//
//  This file is distributed under the same license as the Scilab package.
// =============================================================================

//(titleGraph, xLabel, yLabel, xData, yData, legendeString)
function showMyBench(fileName,varargin)

    rhs=argn(2);
    benchdir = "";
    fontSize = 8;
    smoothCurve = 0;

    if rhs > 1
        benchdir = varargin(1);
    end

    if rhs > 2
        fontSize = varargin(2);
    end

    if rhs > 3
        smoothCurve = varargin(3)
    end

    if rhs > 4
        disp("showMyBench : wrong number of input arguments : 1 to 4 expected.");
    end

    if fileName == "all" then
        tmp = pwd();
        cd(benchdir);
        files = dir("*.h5");
        files = files.name;
        cd(tmp);
    else
        files = fileName;
    end;

    for i = 1:size(files,"*"),

        import_from_hdf5(benchdir + files(i));

        f                   = scf();
        f.figure_size	    = [1920, 1080];
        f.anti_aliasing	    = "16x";
        f.color_map         = jetcolormap(size(yData,'c'));
//        f.color_map         = [144,0,0 ; 0,144,0; 0,0,144; 144,0,144 ; 139,139,0 ; 0,139,139];
        a                   = gca();
        a.title.text        = titleGraph;
        a.title.font_size   = fontSize;
        a.title.font_style  = 2;
        a.x_label.text      = xLabel;
        a.x_label.font_size = 4;
        a.y_label.text      = yLabel;
        a.y_label.font_size = 4;

        if smoothCurve == -1
            smoothCurve = size(xData, '*') / 10;
        end

        for k=1:smoothCurve
            for j=1:size(yData,'c')
                yData(:,j) = [yData(1,j); (yData(1:($-1), j) + yData(2:$, j)) / 2];
            end
        end

        disp(xData)
        disp(yData)

        plot2d(xData,yData);
        legend(f,legendeString,2);
        e = gce();
        e.font_size = 4;
        imgName =  files(i) + '.png';
        xs2png(f, benchdir + imgName);
    end;

endfunction

