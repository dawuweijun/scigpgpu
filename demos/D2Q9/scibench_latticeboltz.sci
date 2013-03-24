// Copyright (C) Adriano Sciacovelli
// Copyright (C) 2006-2008 Jonas Latt
// Copyright (C) 2010 - DIGITEO - Michael Baudin
// Copyright (C) 2010 - DIGITEO - Vincent Lejeune

//
// This file must be used under the terms of the CeCILL.
// This source file is licensed as described in the file COPYING, which
// you should have received as part of this distribution.  The terms
// are also available at
// http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt

function scibench_latticeboltz ( lx , ly , maxT , tPlot )
    // Performs a 2D Lattice Boltzmann simulation with Scilab.
    //
    // Calling Sequence
    //   scibench_latticeboltz ( lx , ly , maxT , tPlot )
    //
    // Parameters
    //   lx : a 1-by-1 matrix of floating point integers, the number of cells in the X direction, must be greater than 1
    //   ly : a 1-by-1 matrix of floating point integers, the number of cells in the Y direction, must be greater than 1
    //   maxT : a 1-by-1 matrix of doubles, the maximum number of iterations, must be greater than 1
    //   tPlot : a 1-by-1 matrix of doubles, the number of iterations between 2 refresh of the graphics, must be greater than 1
    //
    // Description
    //   This function simulates the channel flow past a cylinderical
    //   obstacle, using a Lattice Boltzmann method.
    //
    //   Get the most recent version of this file on LBMethod.org:
    //   http://www.lbmethod.org/_media/numerics:cylinder.m
    //
    //   Original implementaion of Zou/He boundary condition by
    //   Adriano Sciacovelli (see example "cavity.m").
    //   The script was translated to scilab language by Vincent Lejeune.
    //   The performance was improved by Michael Baudin, using vectorized statements.
    //
    // Examples
    // lx     = 400;      // number of cells in x-direction
    // ly     = 100;      // number of cells in y-direction
    // tPlot  = 50;       // cycles
    // maxT   = 1000;     // total number of iterations
    // scf();
    // tic();
    // scibench_latticeboltz ( lx , ly , maxT , tPlot );
    // t = toc();
    // mprintf("lx=%d, ly=%d, maxT=%d, t=%.3f\n",lx,ly,maxT,t);
    //
    // Bibliography
    // http://www.lbmethod.org/_media/numerics:cylinder.m
    //
    // Authors
    // Copyright (C) Adriano Sciacovelli
    // Copyright (C) 2006-2008 - Jonas Latt (jonas@lbmethod.org)
    // Copyright (C) 2010 - DIGITEO - Michael Baudin
    // Copyright (C) 2010 - DIGITEO - Vincent Lejeune

    tic();

    [lhs, rhs] = argn()
    apifun_checkrhs ( "scibench_latticeboltz" , rhs , 4 )
    apifun_checklhs ( "scibench_latticeboltz" , lhs , 0:2 )
    //
    // Check Type
    apifun_checktype ( "scibench_latticeboltz" , lx , "lx" , 1 , "constant" )
    apifun_checktype ( "scibench_latticeboltz" , ly , "ly" , 2 , "constant" )
    apifun_checktype ( "scibench_latticeboltz" , maxT , "maxT" , 3 , "constant" )
    apifun_checktype ( "scibench_latticeboltz" , tPlot , "tPlot" , 4 , "constant" )
    //
    // Check Size
    apifun_checkscalar ( "scibench_latticeboltz" , lx , "lx" , 1 )
    apifun_checkscalar ( "scibench_latticeboltz" , ly , "ly" , 2 )
    apifun_checkscalar ( "scibench_latticeboltz" , maxT , "maxT" , 3 )
    apifun_checkscalar ( "scibench_latticeboltz" , tPlot , "tPlot" , 4 )
    //
    // Check Content
    apifun_checkgreq ( "scibench_latticeboltz" , lx , "lx" , 1 , 1 )
    apifun_checkgreq ( "scibench_latticeboltz" , ly , "ly" , 2 , 1 )
    apifun_checkgreq ( "scibench_latticeboltz" , maxT , "maxT" , 3 , 1 )
    apifun_checkgreq ( "scibench_latticeboltz" , tPlot , "tPlot" , 4 , 1 )
    // TODO : check that these are floating point integers

    function vf = field_zeros(m,n)
        vf = list();
        for i = 1 : 9
            vf(i)=zeros(m,n);
        end
    endfunction

    function s = field_sum(f)
        [m,n] = size(f(1))
        s = zeros(m,n)
        for i = 1 : 9
            s = s + f(i)
        end
    endfunction

    function ret=cs_MB(A,m1,m2)
        [m,n]=size(A);
        rettmp=zeros(m,n);
        ret=zeros(m,n);
        id=pmodulo((1:m)-m1-1,m)+1;
        rettmp(1:m,:)=A(id,:);
        jd=pmodulo((1:n)-m2-1,n)+1;
        ret(:,1:n)=rettmp(:,jd);
    endfunction

    // GENERAL FLOW CONSTANTS
    obst_x = lx/5+1;   // position of the cylinder; (exact
    obst_y = ly/2+3;   // y-symmetry is avoided)
    obst_r = ly/10+1;  // radius of the cylinder
    uMax   = 0.1;      // maximum velocity of Poiseuille inflow
    Re     = 100;      // Reynolds number
    nu     = uMax * 2.*obst_r / Re;  // kinematic viscosity
    omega  = 1. / (3*nu+1./2.);      // relaxation parameter

    // D2Q9 LATTICE CONSTANTS
    t  = [4/9, 1/9,1/9,1/9,1/9, 1/36,1/36,1/36,1/36];
    cx = [  0,   1,  0, -1,  0,    1,  -1,  -1,   1];
    cy = [  0,   0,  1,  0, -1,    1,   1,  -1,  -1];
    opp = [ 1,   4,  5,  2,  3,    8,   9,   6,   7];
    col = [2:(ly-1)];
    in  = 1;   // position of inlet
    out = lx;  // position of outlet

    [y,x] = meshgrid(1:ly,1:lx); // get coordinate of matrix indices

    obst = ...                   // Location of cylinder
    (x-obst_x).^2 + (y-obst_y).^2 <= obst_r.^2;
    obst(:,[1,ly]) = 1;    // Location of top/bottom boundary
    bbRegion = find(obst); // Boolean mask for bounce-back cells

    // INITIAL CONDITION: Poiseuille profile at equilibrium
    L = ly-2; y_phys = y-1.5;
    ux = 4 * uMax / (L*L) * (y_phys.*L-y_phys.*y_phys);
    uy = zeros(lx,ly);
    rho = 1;
    fIn  = field_zeros(lx,ly);
    [fIn1,fIn2,fIn3,fIn4,fIn5,fIn6,fIn7,fIn8,fIn9]=fIn(:);
    fEq  = field_zeros(lx,ly);
    [fEq1,fEq2,fEq3,fEq4,fEq5,fEq6,fEq7,fEq8,fEq9]=fEq(:);
    fOut = field_zeros(lx,ly);
    [fOut1,fOut2,fOut3,fOut4,fOut5,fOut6,fOut7,fOut8,fOut9]=fOut(:);
    for i=1:9
        cu = 3*(cx(i)*ux+cy(i)*uy);
        execstr("fIn"+string(i)+" = rho .* t(i) .* ( 1 + cu + 1/2*(cu.*cu) - 3/2*(ux.^2+uy.^2) )" );
    end

    body_MACROstep = [];
    body_MACROstep(1) = "rho = zeros(lx,ly)";
    body_MACROstep(2) = "ux  = zeros(lx,ly)";
    body_MACROstep(3) = "uy  = zeros(lx,ly)";
    for i = 1 : 9
        body_MACROstep(1) = body_MACROstep(1) + "+ fIn" + string(i)
        body_MACROstep(2) = body_MACROstep(2) + "+ cx("+string(i)+")*fIn"+string(i)
        body_MACROstep(3) = body_MACROstep(3) + "+ cy("+string(i)+")*fIn"+string(i)
    end

    body_COLLISIONstep = [];
    for i=1:9
        body_COLLISIONstep($+1)= "cu = 3*(cx("+string(i)+")*ux+cy("+string(i)+")*uy);";
        body_COLLISIONstep($+1)= "fEq"+string(i)+" = rho .* t("+string(i)+") .*( 1 + cu + 1/2*(cu.*cu)  - 3/2*(ux.^2+uy.^2) );" ;
        body_COLLISIONstep($+1)= "fOut"+string(i)+" = fIn"+string(i)+" - omega .* (fIn"+string(i)+"-fEq"+string(i)+");";
    end

    body_OBSTACLEstep = [];
    for i=1:9
        body_OBSTACLEstep($+1) = "fOut"+string(i)+"(bbRegion) = fIn"+string(opp(i))+"(bbRegion);";
    end

    body_STREAMINGstep = [];
    for i=1:9
        body_STREAMINGstep($+1) = "fIn"+string(i)+" = cs_MB(fOut"+string(i)+",cx("+string(i)+"),cy("+string(i)+"));";
    end

    h = scf();
    firstplot = %t
    e=gce();
    //
    // MAIN LOOP (TIME CYCLES)
    //
    for cycle = 1:maxT
        if is_handle_valid(e) then
            //
            //     // MACROSCOPIC VARIABLES
            execstr(body_MACROstep);
            ux=ux./rho;
            uy=uy./rho;

            // MACROSCOPIC (DIRICHLET) BOUNDARY CONDITIONS
            // Inlet: Poiseuille profile
            y_phys = col-1.5;
            ux(in,col) = 4 * uMax / (L*L) * (y_phys.*L-y_phys.*y_phys);
            uy(in,col) = 0;
            tmp = fIn1(in,col) + fIn3(in,col) + fIn5(in,col) + 2 * (fIn4(in,col) + fIn7(in,col) + fIn8(in,col));
            rho(in,col) = ones(1,ly-2) ./ (1-ux(in,col)) .* tmp;
            // Outlet: Constant pressure
            rho(out,col) = 1;
            tout = fIn1(out,col) + fIn3(out,col) + fIn5(out,col) + 2 * (fIn2(out,col) + fIn6(out,col) + fIn9(out,col));
            ux(out,col) = -ones(1,ly-2) + ones(1,ly-2) ./ (rho(out,col)) .* tout;
            uy(out,col)  = 0;

            // MICROSCOPIC BOUNDARY CONDITIONS: INLET (Zou/He BC)
            fIn2(in,col) = fIn4(in,col) + 2/3*rho(in,col).*ux(in,col);
            fIn6(in,col) = fIn8(in,col) + 1/2*(fIn5(in,col)-fIn3(in,col)) + 1/2*rho(in,col).*uy(in,col) + 1/6*rho(in,col).*ux(in,col);
            fIn9(in,col) = fIn7(in,col) + 1/2*(fIn3(in,col)-fIn5(in,col)) - 1/2*rho(in,col).*uy(in,col) + 1/6*rho(in,col).*ux(in,col);

            // MICROSCOPIC BOUNDARY CONDITIONS: OUTLET (Zou/He BC)
            fIn4(out,col) = fIn2(out,col) - 2/3*rho(out,col).*ux(out,col);
            fIn8(out,col) = fIn6(out,col) + 1/2*(fIn3(out,col)-fIn5(out,col)) - 1/2*rho(out,col).*uy(out,col) - 1/6*rho(out,col).*ux(out,col);
            fIn7(out,col) = fIn9(out,col) + 1/2*(fIn5(out,col)-fIn3(out,col)) + 1/2*rho(out,col).*uy(out,col) - 1/6*rho(out,col).*ux(out,col);

            // COLLISION STEP
            execstr(body_COLLISIONstep);

            // OBSTACLE (BOUNCE-BACK)
            execstr(body_OBSTACLEstep);

            // STREAMING STEP
            execstr(body_STREAMINGstep);
            //
            // VISUALIZATION
            if (pmodulo(cycle,tPlot)==0) then
                drawlater();
               // clf();
                u = sqrt(ux.^2+uy.^2);
                u(bbRegion) = %nan;
                img=abs(255*u/max(u));

                xtitle("Cycle ="+string(cycle)+"/"+string(maxT)+" Elapsed time : "+string(toc())+"s","X","Y")
                if ( firstplot ) then
                    h.color_map = jetcolormap(256);
                    Matplot(img');
                    firstplot = %f;
                    e=gce();
                else
	                if is_handle_valid(e) then
		                 e.data=img';
	                else
		                return
                    end
                end
                drawnow();
            end
        end
    end
endfunction

